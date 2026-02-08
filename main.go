package main

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"log"
	"math"
	"net/http"
	"os"
	"path/filepath"
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
	ID         int64            `json:"id"`
	ImageName  string           `json:"image_name"`
	UserLat    float64          `json:"user_lat"`
	UserLng    float64          `json:"user_lng"`
	TopLabel   string           `json:"top_label"`
	TopPct     int              `json:"top_pct"`
	Items      []predictionItem `json:"items"`
	Pins       []pin            `json:"pins"`
	CreatedAt  string           `json:"created_at"`
}

type listing struct {
	ID        int64   `json:"id"`
	Material  string  `json:"material"`
	WeightKg  float64 `json:"weight_kg"`
	Price     float64 `json:"price"`
	Contact   string  `json:"contact"`
	Lat       float64 `json:"lat"`
	Lng       float64 `json:"lng"`
	Status    string  `json:"status"`
	CreatedAt string  `json:"created_at"`
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

type facility struct {
	Name string
	Icon string
	DLat float64
	DLng float64
}

var facilitiesByType = map[string][]facility{
	"Metal":   {{Name: "Metal Factory", Icon: "🏭", DLat: 0.01, DLng: 0.01}, {Name: "Scrap Yard", Icon: "🔩", DLat: -0.01, DLng: 0.008}},
	"Plastic": {{Name: "Plastic Recycling Center", Icon: "♻️", DLat: 0.01, DLng: 0.005}},
	"Paper":   {{Name: "Paper Recycling Mill", Icon: "📄", DLat: -0.008, DLng: 0.01}},
	"E-Waste": {{Name: "Authorized E-Waste Center", Icon: "💻", DLat: 0.01, DLng: -0.01}},
	"Medical": {{Name: "Medical Waste Facility", Icon: "🏥", DLat: -0.01, DLng: -0.008}},
	"Glass":   {{Name: "Glass Recycling Plant", Icon: "🍾", DLat: 0.007, DLng: -0.009}},
	"Organic": {{Name: "Composting Unit", Icon: "🌱", DLat: -0.006, DLng: 0.007}},
}

var labels = []string{"Plastic", "Metal", "Paper", "E-Waste", "Medical", "Glass", "Organic"}

func main() {
	db, err := sql.Open("sqlite", "file:civicly.db?_pragma=foreign_keys(1)")
	if err != nil {
		log.Fatal(err)
	}
	defer db.Close()

	if err := initDB(db); err != nil {
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
`
	_, err := db.Exec(schema)
	return err
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

		items := generatePredictionItems(header.Filename)
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

func handleListings(db *sql.DB) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodGet:
			rows, err := db.Query(`SELECT id, material, weight_kg, price, contact, lat, lng, status, created_at FROM listings WHERE status = 'open' ORDER BY created_at DESC`)
			if err != nil {
				http.Error(w, "database error", http.StatusInternalServerError)
				return
			}
			defer rows.Close()

			var listings []listing
			for rows.Next() {
				var l listing
				var created time.Time
				if err := rows.Scan(&l.ID, &l.Material, &l.WeightKg, &l.Price, &l.Contact, &l.Lat, &l.Lng, &l.Status, &created); err != nil {
					http.Error(w, "database error", http.StatusInternalServerError)
					return
				}
				l.CreatedAt = created.Format(time.RFC3339)
				listings = append(listings, l)
			}
			writeJSON(w, http.StatusOK, listings)
		case http.MethodPost:
			var req createListingRequest
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				http.Error(w, "invalid json", http.StatusBadRequest)
				return
			}
			if strings.TrimSpace(req.Material) == "" || req.WeightKg <= 0 || req.Price <= 0 || strings.TrimSpace(req.Contact) == "" {
				http.Error(w, "missing or invalid listing fields", http.StatusBadRequest)
				return
			}

			res, err := db.Exec(`INSERT INTO listings (material, weight_kg, price, contact, lat, lng, status) VALUES (?, ?, ?, ?, ?, ?, 'open')`, req.Material, req.WeightKg, req.Price, req.Contact, req.Lat, req.Lng)
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
		if len(parts) != 2 || parts[1] != "buy" {
			http.Error(w, "invalid listing action", http.StatusBadRequest)
			return
		}
		listingID, err := strconv.ParseInt(parts[0], 10, 64)
		if err != nil || listingID <= 0 {
			http.Error(w, "invalid listing id", http.StatusBadRequest)
			return
		}

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
		err = tx.QueryRow(`SELECT status FROM listings WHERE id = ?`, listingID).Scan(&status)
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

		if err := tx.Commit(); err != nil {
			http.Error(w, "database error", http.StatusInternalServerError)
			return
		}

		writeJSON(w, http.StatusOK, map[string]any{"message": "purchase successful", "listing_id": listingID})
	}
}

func generatePredictionItems(fileName string) []predictionItem {
	h := fnv.New64a()
	_, _ = h.Write([]byte(strings.ToLower(filepath.Base(fileName))))
	seed := h.Sum64()

	first := int(seed % uint64(len(labels)))
	second := int((seed / 7) % uint64(len(labels)))
	third := int((seed / 11) % uint64(len(labels)))
	if second == first {
		second = (second + 1) % len(labels)
	}
	if third == first || third == second {
		third = (third + 2) % len(labels)
	}

	base := []int{40 + int(seed%16), 25 + int((seed/3)%16), 10 + int((seed/5)%16)}
	total := base[0] + base[1] + base[2]
	pcts := []int{
		int(math.Round(float64(base[0]) * 100 / float64(total))),
		int(math.Round(float64(base[1]) * 100 / float64(total))),
	}
	pcts = append(pcts, 100-pcts[0]-pcts[1])

	return []predictionItem{
		{Label: labels[first], Pct: pcts[0]},
		{Label: labels[second], Pct: pcts[1]},
		{Label: labels[third], Pct: pcts[2]},
	}
}

func buildPins(lat, lng float64, items []predictionItem) []pin {
	pins := []pin{{Name: "Your Location", Lat: lat, Lng: lng, Icon: "📍", PinType: "user"}}
	for _, it := range items {
		for _, f := range facilitiesByType[it.Label] {
			pins = append(pins, pin{
				Name:    f.Name,
				Lat:     lat + f.DLat,
				Lng:     lng + f.DLng,
				Icon:    f.Icon,
				PinType: "facility",
			})
		}
	}
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

func writeJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(payload); err != nil {
		fmt.Fprintf(os.Stderr, "failed to write response: %v\n", err)
	}
}
