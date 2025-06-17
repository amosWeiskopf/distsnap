// DistSnap: the superfast string similarity mapper
// Author: Amos Weiskopf
// License: MIT
// Released: 2025-06-17

package main

import (
	"bufio"
	"bytes"
	"crypto/sha1"
	"database/sql"
	"encoding/csv"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"os"
	"runtime"
	"sort"
	"strings"
	"sync"

	_ "modernc.org/sqlite"
)

const defaultModel = "text-embedding-3-large"

func main() {
	csvPath := flag.String("csv", "cosine_output.csv", "CSV output file path")
	kFlag := flag.Int("k", 0, "Choose number of clusters (0 = prompt)")
	modelFlag := flag.String("model", defaultModel, "OpenAI embedding model to use")
	flag.Parse()

	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {log.Fatal("Set OPENAI_API_KEY env var")}

	db, err := sql.Open("sqlite", "embedcache.db")
	if err != nil {log.Fatalf("sqlite open: %v", err)}
	defer db.Close()

	_, err = db.Exec(`CREATE TABLE IF NOT EXISTS embed_cache (
		model TEXT NOT NULL,
		sha1 TEXT NOT NULL,
		embedding TEXT NOT NULL,
		PRIMARY KEY(model, sha1)
	)`)
	if err != nil {log.Fatalf("sqlite schema: %v", err)}

	sc := bufio.NewScanner(os.Stdin)
	mode := prompt(sc, "1: Simple demo (boring)\n2: Paste comma-separated strings (boring)")

	var rawTexts []string
	switch mode {
	case "1": for i := 0; i < 3; i++ {rawTexts = append(rawTexts, prompt(sc, fmt.Sprintf("Enter some text (e.g. Quentin Tarantino or cheesecake) #%d:", i+1)))}
	case "2":
		fmt.Println("Paste comma-separated, as many as you like (tested for 3-1000) (For example: \x1b[1;38;2;173;216;230mdajngo unchained, sam altman, quentin tarantino, ray charles, biology, cosine similarity, origin of the species, marc andreesen, LEGO, set theory, metallica, jerry seinfeld, the simpsons, dire straits  \x1b[0m. Please keep in mind, matrix only prints for <12 strings.")
		if !sc.Scan() {log.Fatalf("Input error: %v", sc.Err())}
		for _, s := range strings.Split(sc.Text(), ",") {
			if trimmed := strings.TrimSpace(s); trimmed != "" {rawTexts = append(rawTexts, trimmed)}
		}
	default: log.Fatalf("Invalid mode: %q", mode) }

	if len(rawTexts) < 2 {log.Fatal("Need at least 2 texts for similarity comparison")}

	if *kFlag <= 0 {
		val := prompt(sc, fmt.Sprintf("How many clusters (1–%d)?", len(rawTexts)))
		kParsed, err := parseIntInRange(val, 1, len(rawTexts))
		if err != nil {log.Fatal(err)}
		*kFlag = kParsed
	}

	estTokens := estimateTotalTokens(rawTexts)
	estCost := float64(estTokens) / 1_000_000.0 * 0.13
	fmt.Printf("\nEstimated token count: %d\nEstimated cost: $%.6f\n", estTokens, estCost)

	if strings.ToLower(prompt(sc, "Proceed? (y/n)")) != "y" {  fmt.Println("Aborted."); return }
	emb, actualTokens := fetchEmbeddings(rawTexts, apiKey, *modelFlag, db)
	simMatrix := computeSimMatrix(emb)

if len(rawTexts) <= 12 {
	fmt.Printf("\nCosine Similarity Matrix (0 = red, 1 = green):\n")
	printHeader(rawTexts)
	topCol := make([]float64, len(simMatrix))
	for j := range simMatrix {
		max := -1.0
		for i := range simMatrix {
			v := simMatrix[i][j]
			if v < 1.0 && v > max && math.Abs(v-1.0) > 1e-6 {max = v}
		}
		topCol[j] = max
	}

	for i, row := range simMatrix {
		fmt.Printf("%-22s", trimLabel(rawTexts[i]))
		for j, v := range row {
			highlight := math.Abs(v-topCol[j]) < 1e-6 && v < 1.0
			fmt.Print(colorCell(v, highlight))
		}
		fmt.Println(resetColor())
	}
}




	saveCSV(*csvPath, rawTexts, simMatrix)
	fmt.Printf("\nSaved matrix to %s\n", *csvPath)
	showTopSimilars(rawTexts, simMatrix, 4)

	actualCost := float64(actualTokens) / 1_000_000.0 * 0.13
	fmt.Printf("\nActual tokens: %d\nActual cost: $%.6f\n", actualTokens, actualCost)

	fmt.Println("\nTop semantic clusters:")
	for i, cluster := range agglomerativeClustering(simMatrix, rawTexts, *kFlag) {fmt.Printf("Cluster %d: %s\n", i+1, strings.Join(cluster, ", "))}
}

func fetchEmbeddings(texts []string, apiKey, model string, db *sql.DB) ([][]float64, int) {
	type item struct {
		text string
		hash string
	}
	items := make([]item, len(texts))
	for i, t := range texts {items[i] = item{text: t, hash: sha1Hex(t)}}

	cached := make(map[string][]float64)
	var missing []item
	for _, it := range items {
		var js string
		err := db.QueryRow(`SELECT embedding FROM embed_cache WHERE model=? AND sha1=?`, model, it.hash).Scan(&js)
		if err == nil {
			var emb []float64
			if err := json.Unmarshal([]byte(js), &emb); err == nil {
				cached[it.hash] = emb
				continue
			}
		}
		missing = append(missing, it)
	}

	var toFetch []string
	for _, m := range missing {toFetch = append(toFetch, m.text)}

	fetched := make(map[string][]float64)
	totalTokens := 0
	if len(toFetch) > 0 {
		body, _ := json.Marshal(map[string]any{"input": toFetch, "model": model})
		req, _ := http.NewRequest("POST", "https://api.openai.com/v1/embeddings", bytes.NewReader(body))
		req.Header.Set("Authorization", "Bearer "+apiKey)
		req.Header.Set("Content-Type", "application/json")

		resp, err := http.DefaultClient.Do(req)
		if err != nil {log.Fatalf("HTTP error: %v", err)}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			b, _ := io.ReadAll(resp.Body)
			log.Fatalf("OpenAI API error: %s", string(b))
		}

		var parsed struct {
			Data  []struct{ Index int; Embedding []float64 } `json:"data"`
			Usage struct{ PromptTokens int }                 `json:"usage"`
		}
		if err := json.NewDecoder(resp.Body).Decode(&parsed); err != nil {log.Fatalf("Decode error: %v", err)}
		totalTokens = parsed.Usage.PromptTokens

		for i, d := range parsed.Data {
			m := missing[i]
			fetched[m.hash] = d.Embedding
		}
	}

	result := make([][]float64, len(texts))
	for i, it := range items {
		if emb, ok := cached[it.hash]; ok {result[i] = emb} else {result[i] = fetched[it.hash]}
	}

	go func() {
		tx, _ := db.Begin()
		stmt, _ := tx.Prepare(`INSERT OR IGNORE INTO embed_cache(model, sha1, embedding) VALUES (?, ?, ?)`)
		defer stmt.Close()
		for h, emb := range fetched {
			js, _ := json.Marshal(emb)
			_, _ = stmt.Exec(model, h, string(js))
		}
		_ = tx.Commit()
	}()

	return result, totalTokens
}

func computeSimMatrix(vecs [][]float64) [][]float64 {
	N := len(vecs)
	norm := make([]float64, N)
	for i, v := range vecs {
		var sum float64
		for _, x := range v {sum += x * x}
		norm[i] = 1 / math.Sqrt(sum+1e-10)
	}

	sim := make([][]float64, N)
	for i := range sim {sim[i] = make([]float64, N)}

	type job struct{ i, j int }
	type result struct{ i, j int; val float64 }

	numWorkers := runtime.NumCPU()
	jobs := make(chan job, N*N)
	results := make(chan result, N*N)

	var wg sync.WaitGroup
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for task := range jobs {
				i, j := task.i, task.j
				var dot float64
				for k := range vecs[i] {dot += vecs[i][k] * vecs[j][k]}
				val := dot * norm[i] * norm[j]
				results <- result{i: i, j: j, val: val}
			}
		}()
	}

	go func() {
		for i := 0; i < N; i++ {
			for j := i; j < N; j++ {jobs <- job{i: i, j: j}}
		}
		close(jobs)
	}()

	go func() {
		wg.Wait()
		close(results)
	}()

	for r := range results {
		sim[r.i][r.j] = r.val
		sim[r.j][r.i] = r.val
	}

	return sim
}

func showTopSimilars(labels []string, sim [][]float64, maxTuple int) {
	if len(labels) < 2 {return}
	if maxTuple > len(labels) {maxTuple = len(labels)}
	fmt.Println("\n--- Top Similarity Tuples ---")

	for size := 2; size <= maxTuple; size++ {
		N := len(labels)
		if N > 50 && size >= 4 {
			fmt.Printf("\n>> Skipping %d-tuples (N=%d) for performance\n", size, N)
			continue
		}

		fmt.Printf("\n>> Top %d-tuples:\n", size)

		type group struct {
			Indices []int
			Score   float64
		}
		var groups []group

		for _, idxs := range combinations(N, size) {
			score := avgPairwiseSim(idxs, sim)
			groups = append(groups, group{Indices: idxs, Score: score})
		}

		sort.Slice(groups, func(i, j int) bool {return groups[i].Score > groups[j].Score})

		for i := 0; i < 10 && i < len(groups); i++ {
			var names []string
			for _, idx := range groups[i].Indices {names = append(names, trimLabel(labels[idx]))}
			fmt.Printf("%-80s : %.6f\n", strings.Join(names, " | "), groups[i].Score)
		}
	}
}

func combinations(n, k int) [][]int {
	var res [][]int
	var comb func(start int, path []int)
	comb = func(start int, path []int) {
		if len(path) == k {
			cp := make([]int, k)
			copy(cp, path)
			res = append(res, cp)
			return
		}
		for i := start; i < n; i++ {comb(i+1, append(path, i))}
	}
	comb(0, []int{})
	return res
}

func avgPairwiseSim(idxs []int, sim [][]float64) float64 {
	var total float64
	var count int
	for i := 0; i < len(idxs); i++ {
		for j := i + 1; j < len(idxs); j++ {
			total += sim[idxs[i]][idxs[j]]
			count++
		}
	}
	if count == 0 {return 0}
	return total / float64(count)
}

func agglomerativeClustering(sim [][]float64, labels []string, k int) [][]string {
	type cluster []int
	N := len(sim)
	clusters := make([]cluster, N)
	for i := range clusters {clusters[i] = cluster{i}}

	if k <= 0 || k > N {k = int(math.Sqrt(float64(N)))}

	for len(clusters) > k {
		bestI, bestJ := -1, -1
		bestScore := -1.0

		for i := 0; i < len(clusters); i++ {
			for j := i + 1; j < len(clusters); j++ {
				score := avgClusterSim(clusters[i], clusters[j], sim)
				if score > bestScore {bestScore, bestI, bestJ = score, i, j}
			}
		}

		merged := append(clusters[bestI], clusters[bestJ]...)
		var newClusters []cluster
		for i, c := range clusters {
			if i != bestI && i != bestJ {newClusters = append(newClusters, c)}
		}
		clusters = append(newClusters, merged)
	}

	var result [][]string
	for _, cl := range clusters {
		var group []string
		for _, i := range cl {group = append(group, labels[i])}
		sort.Strings(group)
		result = append(result, group)
	}
	return result
}

func avgClusterSim(a, b []int, sim [][]float64) float64 {
	var total float64
	for _, i := range a {
		for _, j := range b {total += sim[i][j]}
	}
	return total / float64(len(a)*len(b))
}

func saveCSV(path string, headers []string, matrix [][]float64) {
	f, err := os.Create(path)
	if err != nil {log.Fatalf("CSV create error: %v", err)}
	defer f.Close()

	w := csv.NewWriter(f)
	defer w.Flush()

	_ = w.Write(append([]string{""}, headers...))
	for i := range matrix {
		row := []string{headers[i]}
		for _, val := range matrix[i] {row = append(row, fmt.Sprintf("%.6f", val))}
		_ = w.Write(row)
	}
}

func trimLabel(s string) string {
	s = strings.ReplaceAll(s, "\n", " "); s = strings.TrimSpace(s); if len(s) > 20 {return s[:19] + "…"}
	return s
}

func printHeader(texts []string) {
	fmt.Printf("%-22s", "")
	for _, t := range texts {fmt.Printf("%-22s", trimLabel(t))}
	fmt.Println()
}

func colorCell(sim float64, highlight bool) string {
	if sim < 0 {sim = 0}; if sim > 1 {sim = 1}
	if highlight { return fmt.Sprintf("\x1b[38;2;0;255;0m%-22.6f", sim) } // bright green highlighter
	gray := int(sim * 255); return fmt.Sprintf("\x1b[38;2;%d;%d;%dm%-22.6f", gray, gray, gray, sim) // greyscaler
}



func resetColor() string { return "\x1b[0m" }

func prompt(sc *bufio.Scanner, msg string) string {
	fmt.Println(msg)
	fmt.Print("> ")
	if !sc.Scan() {log.Fatalf("Input error: %v", sc.Err())}
	return strings.TrimSpace(sc.Text())
}

func parseIntInRange(s string, min, max int) (int, error) {
	var val int
	_, err := fmt.Sscanf(s, "%d", &val)
	if err != nil || val < min || val > max {return 0, fmt.Errorf("must be between %d and %d", min, max)}
	return val, nil
}

func estimateTotalTokens(texts []string) int { total := 0; for _, t := range texts {total += int(math.Ceil(float64(len(t)) / 4.0))}
	return total
}

func sha1Hex(s string) string { sum := sha1.Sum([]byte(s)); return hex.EncodeToString(sum[:]) }
