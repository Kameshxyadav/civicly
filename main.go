package main

import (
	"bytes"
	"context"
	"crypto/rand"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"time"

	_ "modernc.org/sqlite"
)

type predictionItem struct {
	Label string `json:"label"`
	Pct   int    `json:"pct"`
}

type pin struct {
	Name    string  `json:"name"`
	Lat     float64 `json:"lat"`
	Lng     float64 `json:"lng"`
	Icon    string  `json:"icon"`
	PinType string  `json:"pin_type"`
}

type predictionResponse struct {
	ID        int64            `json:"id"`
	ImageName string           `json:"image_name"`
	UserLat   float64          `json:"user_lat"`
	UserLng   float64          `json:"user_lng"`
	TopLabel  string           `json:"top_label"`
	TopPct    int              `json:"top_pct"`
	Items     []predictionItem `json:"items"`
	Pins      []pin            `json:"pins"`
	CreatedAt string           `json:"created_at"`
}

type listing struct {
	ID        int64   `json:"id"`
	Material  string  `json:"material"`
	WeightKg  float64 `json:"weight_kg"`
	Price     float64 `json:"price"`
	Contact   string  `json:"contact"`
	Lat       float64 `json:"lat"`
	Lng       float64 `json:"lng"`
	ImageURL  string  `json:"image_url"`
	Status    string  `json:"status"`
	UserID    int64   `json:"user_id"`
	CreatedAt string  `json:"created_at"`
}

type user struct {
	ID        int64  `json:"id"`
	Username  string `json:"username"`
	Password  string `json:"password,omitempty"`
	Role      string `json:"role"`
	CreatedAt string `json:"created_at"`
}

type notification struct {
	ID        int64  `json:"id"`
	UserID    int64  `json:"user_id"`
	Message   string `json:"message"`
	IsRead    bool   `json:"is_read"`
	CreatedAt string `json:"created_at"`
}

type authRequest struct {
	Username string `json:"username"`
	Password string `json:"password"`
}

type createListingRequest struct {
	Material string  `json:"material"`
	WeightKg float64 `json:"weight_kg"`
	Price    float64 `json:"price"`
	Contact  string  `json:"contact"`
	Lat      float64 `json:"lat"`
	Lng      float64 `json:"lng"`
}

type buyRequest struct {
	BuyerContact string `json:"buyer_contact"`
}

type heatmapPoint struct {
	Lat   float64 `json:"lat"`
	Lng   float64 `json:"lng"`
	Label string  `json:"label"`
	Count int     `json:"count"`
}

type facility struct {
	Name string
	Icon string
	DLat float64
	DLng float64
}

var inferencePython = "python3"

var facilitiesByType = map[string][]facility{
	"cardboard": {{Name: "Paper Recovery Facility", Icon: "📦", DLat: 0.007, DLng: 0.009}},
	"e-waste":   {{Name: "Authorized E-Waste Center", Icon: "💻", DLat: 0.01, DLng: -0.01}},
	"glass":     {{Name: "Glass Recycling Plant", Icon: "🍾", DLat: 0.007, DLng: -0.009}},
	"medical":   {{Name: "Medical Waste Facility", Icon: "🏥", DLat: -0.01, DLng: -0.008}},
	"metal":     {{Name: "Metal Factory", Icon: "🏭", DLat: 0.01, DLng: 0.01}, {Name: "Scrap Yard", Icon: "🔩", DLat: -0.01, DLng: 0.008}},
	"organic":   {{Name: "Composting Unit", Icon: "🌱", DLat: -0.006, DLng: 0.007}},
	"paper":     {{Name: "Paper Recycling Mill", Icon: "📄", DLat: -0.008, DLng: 0.01}},
	"plastic":   {{Name: "Plastic Recycling Center", Icon: "♻️", DLat: 0.01, DLng: 0.005}},
}

func main() {
	pythonCmd, err := resolveInferencePython()
	if err != nil {
		log.Printf("warning: model runtime unavailable: %v", err)
	} else {
		inferencePython = pythonCmd
		log.Printf("model runtime: %s", inferencePython)
	}

	db, err := sql.Open("sqlite", "file:civicly.db?_pragma=foreign_keys(1)")
	if err != nil {
		log.Fatal(err)
	}
	defer db.Close()

	if err := initDB(db); err != nil {
		log.Fatal(err)
	}

	if err := os.MkdirAll("uploads", 0o755); err != nil {
		log.Fatal(err)
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/api/health", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
	})
	mux.HandleFunc("/api/predict", withMethod(http.MethodPost, handlePredict(db)))
	mux.HandleFunc("/api/predictions/", withMethod(http.MethodGet, handleGetPrediction(db)))
	mux.HandleFunc("/api/listings", handleListings(db))
	mux.HandleFunc("/api/listings/", handleListingAction(db))
	mux.HandleFunc("/api/register", withMethod(http.MethodPost, handleRegister(db)))
	mux.HandleFunc("/api/login", withMethod(http.MethodPost, handleLogin(db)))
	mux.HandleFunc("/api/logout", withMethod(http.MethodPost, handleLogout(db)))
	mux.HandleFunc("/api/me", withMethod(http.MethodGet, handleMe(db)))
	mux.HandleFunc("/api/notifications", withMethod(http.MethodGet, handleGetNotifications(db)))
	mux.HandleFunc("/api/notifications/read", withMethod(http.MethodPost, handleMarkNotificationsRead(db)))
	mux.HandleFunc("/api/heatmap", withMethod(http.MethodGet, handleHeatmap(db)))
	mux.HandleFunc("/api/admin/users", withMethod(http.MethodGet, handleAdminUsers(db)))
	mux.HandleFunc("/api/admin/listings", withMethod(http.MethodGet, handleAdminListings(db)))
	mux.Handle("/uploads/", http.StripPrefix("/uploads/", http.FileServer(http.Dir("uploads"))))
	mux.HandleFunc("/", serveStatic)

	addr := ":8080"
	if v := strings.TrimSpace(os.Getenv("PORT")); v != "" {
		addr = ":" + v
	}

	log.Printf("civicLy backend running on http://localhost%s", addr)
	if err := http.ListenAndServe(addr, logMiddleware(mux)); err != nil {
		log.Fatal(err)
	}
}

func serveStatic(w http.ResponseWriter, r *http.Request) {
	if strings.HasPrefix(r.URL.Path, "/api/") {
		http.NotFound(w, r)
		return
	}
	if r.URL.Path == "/" {
		http.ServeFile(w, r, "index.html")
		return
	}
	http.FileServer(http.Dir(".")).ServeHTTP(w, r)
}

func logMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		next.ServeHTTP(w, r)
		log.Printf("%s %s (%s)", r.Method, r.URL.Path, time.Since(start).Round(time.Millisecond))
	})
}

func initDB(db *sql.DB) error {
	schema := `
CREATE TABLE IF NOT EXISTS predictions (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	image_name TEXT NOT NULL,
	user_lat REAL NOT NULL,
	user_lng REAL NOT NULL,
	top_label TEXT NOT NULL,
	top_pct INTEGER NOT NULL,
	created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS prediction_items (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	prediction_id INTEGER NOT NULL,
	label TEXT NOT NULL,
	pct INTEGER NOT NULL,
	FOREIGN KEY(prediction_id) REFERENCES predictions(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS map_pins (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	prediction_id INTEGER NOT NULL,
	name TEXT NOT NULL,
	lat REAL NOT NULL,
	lng REAL NOT NULL,
	icon TEXT NOT NULL,
	pin_type TEXT NOT NULL,
	FOREIGN KEY(prediction_id) REFERENCES predictions(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS listings (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	material TEXT NOT NULL,
	weight_kg REAL NOT NULL,
	price REAL NOT NULL,
	contact TEXT NOT NULL,
	lat REAL NOT NULL,
	lng REAL NOT NULL,
	image_url TEXT NOT NULL DEFAULT '',
	status TEXT NOT NULL DEFAULT 'open',
	created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS trades (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	listing_id INTEGER NOT NULL,
	buyer_contact TEXT NOT NULL,
	created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
	FOREIGN KEY(listing_id) REFERENCES listings(id)
);

CREATE TABLE IF NOT EXISTS users (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	username TEXT NOT NULL UNIQUE,
	password TEXT NOT NULL,
	role TEXT NOT NULL DEFAULT 'user',
	created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS sessions (
	token TEXT PRIMARY KEY,
	user_id INTEGER NOT NULL,
	created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
	FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS notifications (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	user_id INTEGER NOT NULL,
	message TEXT NOT NULL,
	is_read INTEGER NOT NULL DEFAULT 0,
	created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
	FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);
`
	_, err := db.Exec(schema)
	if err != nil {
		return err
	}

	// Migrations for existing databases.
	db.Exec(`ALTER TABLE listings ADD COLUMN image_url TEXT NOT NULL DEFAULT ''`)
	db.Exec(`ALTER TABLE listings ADD COLUMN user_id INTEGER NOT NULL DEFAULT 0`)

	// Seed admin user if not exists.
	var count int
	db.QueryRow(`SELECT COUNT(*) FROM users WHERE username = 'admin'`).Scan(&count)
	if count == 0 {
		db.Exec(`INSERT INTO users (username, password, role) VALUES ('admin', 'admin123', 'admin')`)
	}

	return nil
}

func handlePredict(db *sql.DB) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if err := r.ParseMultipartForm(10 << 20); err != nil {
			http.Error(w, "invalid multipart form", http.StatusBadRequest)
			return
		}

		file, header, err := r.FormFile("image")
		if err != nil {
			http.Error(w, "image is required", http.StatusBadRequest)
			return
		}
		defer file.Close()

		lat, err := strconv.ParseFloat(r.FormValue("lat"), 64)
		if err != nil {
			http.Error(w, "valid lat is required", http.StatusBadRequest)
			return
		}
		lng, err := strconv.ParseFloat(r.FormValue("lng"), 64)
		if err != nil {
			http.Error(w, "valid lng is required", http.StatusBadRequest)
			return
		}

		ext := filepath.Ext(header.Filename)
		tmpFile, err := os.CreateTemp("", "civicly-upload-*"+ext)
		if err != nil {
			http.Error(w, "failed to process upload", http.StatusInternalServerError)
			return
		}
		tmpPath := tmpFile.Name()
		defer os.Remove(tmpPath)

		if _, err := io.Copy(tmpFile, file); err != nil {
			_ = tmpFile.Close()
			http.Error(w, "failed to process upload", http.StatusInternalServerError)
			return
		}
		if err := tmpFile.Close(); err != nil {
			http.Error(w, "failed to process upload", http.StatusInternalServerError)
			return
		}

		inferCtx, cancel := context.WithTimeout(r.Context(), 90*time.Second)
		defer cancel()
		items, err := runModelInference(inferCtx, tmpPath)
		if err != nil {
			log.Printf("inference error: %v", err)
			http.Error(w, "model inference failed: "+err.Error(), http.StatusInternalServerError)
			return
		}

		top := items[0]
		for _, it := range items[1:] {
			if it.Pct > top.Pct {
				top = it
			}
		}

		pins := buildPins(lat, lng, items)

		ctx := context.Background()
		tx, err := db.BeginTx(ctx, nil)
		if err != nil {
			http.Error(w, "database error", http.StatusInternalServerError)
			return
		}
		defer tx.Rollback()

		res, err := tx.ExecContext(ctx, `INSERT INTO predictions (image_name, user_lat, user_lng, top_label, top_pct) VALUES (?, ?, ?, ?, ?)`, header.Filename, lat, lng, top.Label, top.Pct)
		if err != nil {
			http.Error(w, "database error", http.StatusInternalServerError)
			return
		}
		predictionID, err := res.LastInsertId()
		if err != nil {
			http.Error(w, "database error", http.StatusInternalServerError)
			return
		}

		for _, it := range items {
			if _, err := tx.ExecContext(ctx, `INSERT INTO prediction_items (prediction_id, label, pct) VALUES (?, ?, ?)`, predictionID, it.Label, it.Pct); err != nil {
				http.Error(w, "database error", http.StatusInternalServerError)
				return
			}
		}

		for _, p := range pins {
			if _, err := tx.ExecContext(ctx, `INSERT INTO map_pins (prediction_id, name, lat, lng, icon, pin_type) VALUES (?, ?, ?, ?, ?, ?)`, predictionID, p.Name, p.Lat, p.Lng, p.Icon, p.PinType); err != nil {
				http.Error(w, "database error", http.StatusInternalServerError)
				return
			}
		}

		if err := tx.Commit(); err != nil {
			http.Error(w, "database error", http.StatusInternalServerError)
			return
		}

		writeJSON(w, http.StatusCreated, predictionResponse{
			ID:        predictionID,
			ImageName: header.Filename,
			UserLat:   lat,
			UserLng:   lng,
			TopLabel:  top.Label,
			TopPct:    top.Pct,
			Items:     items,
			Pins:      pins,
			CreatedAt: time.Now().Format(time.RFC3339),
		})
	}
}

func handleGetPrediction(db *sql.DB) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		idStr := strings.TrimPrefix(r.URL.Path, "/api/predictions/")
		id, err := strconv.ParseInt(idStr, 10, 64)
		if err != nil || id <= 0 {
			http.Error(w, "invalid prediction id", http.StatusBadRequest)
			return
		}

		var out predictionResponse
		var created time.Time
		err = db.QueryRow(`SELECT id, image_name, user_lat, user_lng, top_label, top_pct, created_at FROM predictions WHERE id = ?`, id).
			Scan(&out.ID, &out.ImageName, &out.UserLat, &out.UserLng, &out.TopLabel, &out.TopPct, &created)
		if err == sql.ErrNoRows {
			http.Error(w, "prediction not found", http.StatusNotFound)
			return
		}
		if err != nil {
			http.Error(w, "database error", http.StatusInternalServerError)
			return
		}
		out.CreatedAt = created.Format(time.RFC3339)

		itemRows, err := db.Query(`SELECT label, pct FROM prediction_items WHERE prediction_id = ? ORDER BY pct DESC`, id)
		if err != nil {
			http.Error(w, "database error", http.StatusInternalServerError)
			return
		}
		defer itemRows.Close()

		for itemRows.Next() {
			var it predictionItem
			if err := itemRows.Scan(&it.Label, &it.Pct); err != nil {
				http.Error(w, "database error", http.StatusInternalServerError)
				return
			}
			out.Items = append(out.Items, it)
		}

		pinRows, err := db.Query(`SELECT name, lat, lng, icon, pin_type FROM map_pins WHERE prediction_id = ?`, id)
		if err != nil {
			http.Error(w, "database error", http.StatusInternalServerError)
			return
		}
		defer pinRows.Close()
		for pinRows.Next() {
			var p pin
			if err := pinRows.Scan(&p.Name, &p.Lat, &p.Lng, &p.Icon, &p.PinType); err != nil {
				http.Error(w, "database error", http.StatusInternalServerError)
				return
			}
			out.Pins = append(out.Pins, p)
		}

		writeJSON(w, http.StatusOK, out)
	}
}

func handleHeatmap(db *sql.DB) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		rows, err := db.Query(`
			SELECT user_lat, user_lng, top_label, COUNT(*) as cnt
			FROM predictions
			GROUP BY user_lat, user_lng, top_label
			ORDER BY cnt DESC
		`)
		if err != nil {
			http.Error(w, "database error", http.StatusInternalServerError)
			return
		}
		defer rows.Close()

		var points []heatmapPoint
		for rows.Next() {
			var p heatmapPoint
			if err := rows.Scan(&p.Lat, &p.Lng, &p.Label, &p.Count); err != nil {
				http.Error(w, "database error", http.StatusInternalServerError)
				return
			}
			points = append(points, p)
		}
		if points == nil {
			points = []heatmapPoint{}
		}
		writeJSON(w, http.StatusOK, points)
	}
}

func handleListings(db *sql.DB) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodGet:
			rows, err := db.Query(`SELECT id, material, weight_kg, price, contact, lat, lng, image_url, status, user_id, created_at FROM listings WHERE status = 'open' ORDER BY created_at DESC`)
			if err != nil {
				http.Error(w, "database error", http.StatusInternalServerError)
				return
			}
			defer rows.Close()

			var listings []listing
			for rows.Next() {
				var l listing
				var created time.Time
				if err := rows.Scan(&l.ID, &l.Material, &l.WeightKg, &l.Price, &l.Contact, &l.Lat, &l.Lng, &l.ImageURL, &l.Status, &l.UserID, &created); err != nil {
					http.Error(w, "database error", http.StatusInternalServerError)
					return
				}
				l.CreatedAt = created.Format(time.RFC3339)
				listings = append(listings, l)
			}
			writeJSON(w, http.StatusOK, listings)
		case http.MethodPost:
			sessionUser, err := getSessionUser(db, r)
			if err != nil {
				http.Error(w, "login required", http.StatusUnauthorized)
				return
			}

			if err := r.ParseMultipartForm(10 << 20); err != nil {
				http.Error(w, "invalid multipart form", http.StatusBadRequest)
				return
			}

			material := strings.TrimSpace(r.FormValue("material"))
			weightKg, _ := strconv.ParseFloat(r.FormValue("weight_kg"), 64)
			price, _ := strconv.ParseFloat(r.FormValue("price"), 64)
			contact := strings.TrimSpace(r.FormValue("contact"))
			lat, _ := strconv.ParseFloat(r.FormValue("lat"), 64)
			lng, _ := strconv.ParseFloat(r.FormValue("lng"), 64)

			if material == "" || weightKg <= 0 || price <= 0 || contact == "" {
				http.Error(w, "missing or invalid listing fields", http.StatusBadRequest)
				return
			}

			var imageURL string
			file, header, err := r.FormFile("image")
			if err == nil {
				defer file.Close()
				ext := filepath.Ext(header.Filename)
				filename := fmt.Sprintf("%d%s", time.Now().UnixNano(), ext)
				dst, err := os.Create(filepath.Join("uploads", filename))
				if err != nil {
					http.Error(w, "failed to save image", http.StatusInternalServerError)
					return
				}
				defer dst.Close()
				if _, err := io.Copy(dst, file); err != nil {
					http.Error(w, "failed to save image", http.StatusInternalServerError)
					return
				}
				imageURL = "/uploads/" + filename
			}

			res, err := db.Exec(`INSERT INTO listings (material, weight_kg, price, contact, lat, lng, image_url, status, user_id) VALUES (?, ?, ?, ?, ?, ?, ?, 'open', ?)`, material, weightKg, price, contact, lat, lng, imageURL, sessionUser.ID)
			if err != nil {
				http.Error(w, "database error", http.StatusInternalServerError)
				return
			}
			id, _ := res.LastInsertId()
			writeJSON(w, http.StatusCreated, map[string]any{"id": id, "message": "listing created"})
		default:
			w.Header().Set("Allow", "GET, POST")
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		}
	}
}

func handleListingAction(db *sql.DB) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			w.Header().Set("Allow", "POST")
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}

		trimmed := strings.Trim(strings.TrimPrefix(r.URL.Path, "/api/listings/"), "/")
		parts := strings.Split(trimmed, "/")
		if len(parts) != 2 || (parts[1] != "buy" && parts[1] != "sold") {
			http.Error(w, "invalid listing action", http.StatusBadRequest)
			return
		}
		listingID, err := strconv.ParseInt(parts[0], 10, 64)
		if err != nil || listingID <= 0 {
			http.Error(w, "invalid listing id", http.StatusBadRequest)
			return
		}

		action := parts[1]

		if action == "sold" {
			sessionUser, err := getSessionUser(db, r)
			if err != nil {
				http.Error(w, "login required", http.StatusUnauthorized)
				return
			}
			var ownerID int64
			var status string
			err = db.QueryRow(`SELECT user_id, status FROM listings WHERE id = ?`, listingID).Scan(&ownerID, &status)
			if err == sql.ErrNoRows {
				http.Error(w, "listing not found", http.StatusNotFound)
				return
			}
			if err != nil {
				http.Error(w, "database error", http.StatusInternalServerError)
				return
			}
			if ownerID != sessionUser.ID {
				http.Error(w, "forbidden", http.StatusForbidden)
				return
			}
			if status != "open" {
				http.Error(w, "listing already sold", http.StatusConflict)
				return
			}
			if _, err := db.Exec(`UPDATE listings SET status = 'sold' WHERE id = ?`, listingID); err != nil {
				http.Error(w, "database error", http.StatusInternalServerError)
				return
			}
			writeJSON(w, http.StatusOK, map[string]any{"message": "listing marked as sold", "listing_id": listingID})
			return
		}

		// action == "buy"
		var req buyRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "invalid json", http.StatusBadRequest)
			return
		}
		if strings.TrimSpace(req.BuyerContact) == "" {
			http.Error(w, "buyer contact is required", http.StatusBadRequest)
			return
		}

		tx, err := db.BeginTx(r.Context(), nil)
		if err != nil {
			http.Error(w, "database error", http.StatusInternalServerError)
			return
		}
		defer tx.Rollback()

		var status string
		var material string
		var sellerUserID int64
		err = tx.QueryRow(`SELECT status, material, user_id FROM listings WHERE id = ?`, listingID).Scan(&status, &material, &sellerUserID)
		if err == sql.ErrNoRows {
			http.Error(w, "listing not found", http.StatusNotFound)
			return
		}
		if err != nil {
			http.Error(w, "database error", http.StatusInternalServerError)
			return
		}
		if status != "open" {
			http.Error(w, "listing already sold", http.StatusConflict)
			return
		}

		if _, err := tx.Exec(`INSERT INTO trades (listing_id, buyer_contact) VALUES (?, ?)`, listingID, req.BuyerContact); err != nil {
			http.Error(w, "database error", http.StatusInternalServerError)
			return
		}
		if _, err := tx.Exec(`UPDATE listings SET status = 'sold' WHERE id = ?`, listingID); err != nil {
			http.Error(w, "database error", http.StatusInternalServerError)
			return
		}

		// Notify the seller if they have an account.
		if sellerUserID > 0 {
			msg := fmt.Sprintf("Your %s listing was purchased! Buyer contact: %s", material, req.BuyerContact)
			tx.Exec(`INSERT INTO notifications (user_id, message) VALUES (?, ?)`, sellerUserID, msg)
		}

		if err := tx.Commit(); err != nil {
			http.Error(w, "database error", http.StatusInternalServerError)
			return
		}

		writeJSON(w, http.StatusOK, map[string]any{"message": "purchase successful", "listing_id": listingID})
	}
}

type modelInferenceOutput struct {
	Items []struct {
		Label string  `json:"label"`
		Score float64 `json:"score"`
	} `json:"items"`
}

func runModelInference(ctx context.Context, imagePath string) ([]predictionItem, error) {
	cmd := exec.CommandContext(ctx, inferencePython, "predict.py", "--image", imagePath, "--top-k", "3")
	var stdout bytes.Buffer
	var stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	err := cmd.Run()
	if err != nil {
		return nil, fmt.Errorf("python inference failed: %w: %s", err, strings.TrimSpace(stderr.String()))
	}

	var raw modelInferenceOutput
	if err := json.Unmarshal(stdout.Bytes(), &raw); err != nil {
		return nil, fmt.Errorf("invalid inference response: %w", err)
	}
	if len(raw.Items) == 0 {
		return nil, fmt.Errorf("inference returned no classes")
	}

	// Convert probabilities to rounded percentages while preserving total=100.
	total := 0
	items := make([]predictionItem, 0, len(raw.Items))
	for _, it := range raw.Items {
		pct := int(it.Score * 100)
		total += pct
		items = append(items, predictionItem{
			Label: humanizeLabel(it.Label),
			Pct:   pct,
		})
	}
	if len(items) > 0 && total < 100 {
		items[0].Pct += 100 - total
	}

	// Drop items that rounded to 0% so they don't clutter results or the map.
	filtered := make([]predictionItem, 0, len(items))
	for _, it := range items {
		if it.Pct > 0 {
			filtered = append(filtered, it)
		}
	}
	return filtered, nil
}

func resolveInferencePython() (string, error) {
	candidates := []string{
		strings.TrimSpace(os.Getenv("CIVICLY_PYTHON")),
		".venv/bin/python3",
		"venv/bin/python3",
		"python3",
	}

	var errs []error
	for _, candidate := range candidates {
		if candidate == "" {
			continue
		}

		path, err := resolveExecutable(candidate)
		if err != nil {
			errs = append(errs, fmt.Errorf("%s: %w", candidate, err))
			continue
		}

		ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		cmd := exec.CommandContext(ctx, path, "-c", "import torch, transformers, PIL")
		out, checkErr := cmd.CombinedOutput()
		cancel()
		if checkErr != nil {
			errs = append(errs, fmt.Errorf("%s missing deps: %s", path, strings.TrimSpace(string(out))))
			continue
		}
		return path, nil
	}

	if len(errs) == 0 {
		return "", errors.New("no python runtime candidates found")
	}

	msgs := make([]string, 0, len(errs))
	for _, e := range errs {
		msgs = append(msgs, e.Error())
	}
	return "", errors.New(strings.Join(msgs, "; "))
}

func resolveExecutable(name string) (string, error) {
	if strings.Contains(name, "/") {
		info, err := os.Stat(name)
		if err != nil {
			return "", err
		}
		if info.IsDir() {
			return "", fmt.Errorf("is a directory")
		}
		return name, nil
	}
	return exec.LookPath(name)
}

func humanizeLabel(label string) string {
	key := normalizeLabel(label)
	if key == "e-waste" {
		return "E-Waste"
	}

	parts := strings.FieldsFunc(key, func(r rune) bool { return r == '-' || r == '_' || r == ' ' })
	if len(parts) == 0 {
		return label
	}

	for i := range parts {
		if parts[i] == "" {
			continue
		}
		parts[i] = strings.ToUpper(parts[i][:1]) + parts[i][1:]
	}
	return strings.Join(parts, " ")
}

func normalizeLabel(label string) string {
	out := strings.ToLower(strings.TrimSpace(label))
	out = strings.ReplaceAll(out, "_", "-")
	return out
}

func buildPins(lat, lng float64, items []predictionItem) []pin {
	pins := []pin{{Name: "Your Location", Lat: lat, Lng: lng, Icon: "📍", PinType: "user"}}
	for _, it := range items {
		normalized := normalizeLabel(it.Label)
		for _, f := range facilitiesByType[normalized] {
			pins = append(pins, pin{
				Name:    f.Name,
				Lat:     lat + f.DLat,
				Lng:     lng + f.DLng,
				Icon:    f.Icon,
				PinType: "facility",
			})
		}
	}
	slices.SortFunc(pins[1:], func(a, b pin) int { return strings.Compare(a.Name, b.Name) })
	return pins
}

func withMethod(method string, next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != method {
			w.Header().Set("Allow", method)
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		next(w, r)
	}
}

// ───────────────── Auth helpers ─────────────────

func generateToken() (string, error) {
	b := make([]byte, 32)
	if _, err := rand.Read(b); err != nil {
		return "", err
	}
	return hex.EncodeToString(b), nil
}

func getSessionUser(db *sql.DB, r *http.Request) (*user, error) {
	cookie, err := r.Cookie("session")
	if err != nil {
		return nil, errors.New("no session")
	}
	var u user
	var created time.Time
	err = db.QueryRow(`SELECT u.id, u.username, u.role, u.created_at FROM users u JOIN sessions s ON s.user_id = u.id WHERE s.token = ?`, cookie.Value).
		Scan(&u.ID, &u.Username, &u.Role, &created)
	if err != nil {
		return nil, errors.New("invalid session")
	}
	u.CreatedAt = created.Format(time.RFC3339)
	return &u, nil
}

func requireAdmin(db *sql.DB, r *http.Request) (*user, error) {
	u, err := getSessionUser(db, r)
	if err != nil {
		return nil, err
	}
	if u.Role != "admin" {
		return nil, errors.New("admin required")
	}
	return u, nil
}

// ───────────────── Auth handlers ─────────────────

func handleRegister(db *sql.DB) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req authRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "invalid json", http.StatusBadRequest)
			return
		}
		req.Username = strings.TrimSpace(req.Username)
		if req.Username == "" || req.Password == "" {
			http.Error(w, "username and password required", http.StatusBadRequest)
			return
		}

		res, err := db.Exec(`INSERT INTO users (username, password, role) VALUES (?, ?, 'user')`, req.Username, req.Password)
		if err != nil {
			if strings.Contains(err.Error(), "UNIQUE") {
				http.Error(w, "username already taken", http.StatusConflict)
				return
			}
			http.Error(w, "database error", http.StatusInternalServerError)
			return
		}
		userID, _ := res.LastInsertId()

		token, err := generateToken()
		if err != nil {
			http.Error(w, "server error", http.StatusInternalServerError)
			return
		}
		db.Exec(`INSERT INTO sessions (token, user_id) VALUES (?, ?)`, token, userID)

		http.SetCookie(w, &http.Cookie{
			Name:     "session",
			Value:    token,
			Path:     "/",
			HttpOnly: true,
			MaxAge:   7 * 24 * 60 * 60,
		})
		writeJSON(w, http.StatusCreated, map[string]any{"id": userID, "username": req.Username, "role": "user"})
	}
}

func handleLogin(db *sql.DB) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req authRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "invalid json", http.StatusBadRequest)
			return
		}
		req.Username = strings.TrimSpace(req.Username)

		var u user
		var created time.Time
		err := db.QueryRow(`SELECT id, username, password, role, created_at FROM users WHERE username = ?`, req.Username).
			Scan(&u.ID, &u.Username, &u.Password, &u.Role, &created)
		if err != nil || u.Password != req.Password {
			http.Error(w, "invalid credentials", http.StatusUnauthorized)
			return
		}
		u.CreatedAt = created.Format(time.RFC3339)

		token, err := generateToken()
		if err != nil {
			http.Error(w, "server error", http.StatusInternalServerError)
			return
		}
		db.Exec(`INSERT INTO sessions (token, user_id) VALUES (?, ?)`, token, u.ID)

		http.SetCookie(w, &http.Cookie{
			Name:     "session",
			Value:    token,
			Path:     "/",
			HttpOnly: true,
			MaxAge:   7 * 24 * 60 * 60,
		})
		writeJSON(w, http.StatusOK, map[string]any{"id": u.ID, "username": u.Username, "role": u.Role})
	}
}

func handleLogout(db *sql.DB) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		cookie, err := r.Cookie("session")
		if err == nil {
			db.Exec(`DELETE FROM sessions WHERE token = ?`, cookie.Value)
		}
		http.SetCookie(w, &http.Cookie{
			Name:     "session",
			Value:    "",
			Path:     "/",
			HttpOnly: true,
			MaxAge:   -1,
		})
		writeJSON(w, http.StatusOK, map[string]string{"message": "logged out"})
	}
}

func handleMe(db *sql.DB) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		u, err := getSessionUser(db, r)
		if err != nil {
			http.Error(w, "not logged in", http.StatusUnauthorized)
			return
		}
		var unread int
		db.QueryRow(`SELECT COUNT(*) FROM notifications WHERE user_id = ? AND is_read = 0`, u.ID).Scan(&unread)
		writeJSON(w, http.StatusOK, map[string]any{
			"id":       u.ID,
			"username": u.Username,
			"role":     u.Role,
			"unread":   unread,
		})
	}
}

// ───────────────── Notification handlers ─────────────────

func handleGetNotifications(db *sql.DB) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		u, err := getSessionUser(db, r)
		if err != nil {
			http.Error(w, "login required", http.StatusUnauthorized)
			return
		}
		rows, err := db.Query(`SELECT id, user_id, message, is_read, created_at FROM notifications WHERE user_id = ? ORDER BY created_at DESC LIMIT 50`, u.ID)
		if err != nil {
			http.Error(w, "database error", http.StatusInternalServerError)
			return
		}
		defer rows.Close()

		var notifs []notification
		for rows.Next() {
			var n notification
			var created time.Time
			var isReadInt int
			if err := rows.Scan(&n.ID, &n.UserID, &n.Message, &isReadInt, &created); err != nil {
				http.Error(w, "database error", http.StatusInternalServerError)
				return
			}
			n.IsRead = isReadInt != 0
			n.CreatedAt = created.Format(time.RFC3339)
			notifs = append(notifs, n)
		}
		writeJSON(w, http.StatusOK, notifs)
	}
}

func handleMarkNotificationsRead(db *sql.DB) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		u, err := getSessionUser(db, r)
		if err != nil {
			http.Error(w, "login required", http.StatusUnauthorized)
			return
		}
		db.Exec(`UPDATE notifications SET is_read = 1 WHERE user_id = ?`, u.ID)
		writeJSON(w, http.StatusOK, map[string]string{"message": "notifications marked as read"})
	}
}

// ───────────────── Admin handlers ─────────────────

func handleAdminUsers(db *sql.DB) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if _, err := requireAdmin(db, r); err != nil {
			http.Error(w, "forbidden", http.StatusForbidden)
			return
		}
		rows, err := db.Query(`SELECT id, username, password, role, created_at FROM users ORDER BY id`)
		if err != nil {
			http.Error(w, "database error", http.StatusInternalServerError)
			return
		}
		defer rows.Close()

		var users []user
		for rows.Next() {
			var u user
			var created time.Time
			if err := rows.Scan(&u.ID, &u.Username, &u.Password, &u.Role, &created); err != nil {
				http.Error(w, "database error", http.StatusInternalServerError)
				return
			}
			u.CreatedAt = created.Format(time.RFC3339)
			users = append(users, u)
		}
		writeJSON(w, http.StatusOK, users)
	}
}

func handleAdminListings(db *sql.DB) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if _, err := requireAdmin(db, r); err != nil {
			http.Error(w, "forbidden", http.StatusForbidden)
			return
		}
		rows, err := db.Query(`SELECT id, material, weight_kg, price, contact, lat, lng, image_url, status, user_id, created_at FROM listings ORDER BY created_at DESC`)
		if err != nil {
			http.Error(w, "database error", http.StatusInternalServerError)
			return
		}
		defer rows.Close()

		var listings []listing
		for rows.Next() {
			var l listing
			var created time.Time
			if err := rows.Scan(&l.ID, &l.Material, &l.WeightKg, &l.Price, &l.Contact, &l.Lat, &l.Lng, &l.ImageURL, &l.Status, &l.UserID, &created); err != nil {
				http.Error(w, "database error", http.StatusInternalServerError)
				return
			}
			l.CreatedAt = created.Format(time.RFC3339)
			listings = append(listings, l)
		}
		writeJSON(w, http.StatusOK, listings)
	}
}

func writeJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(payload); err != nil {
		fmt.Fprintf(os.Stderr, "failed to write response: %v\n", err)
	}
}
