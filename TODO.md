# goto-gh - Plan d'Amélioration

> Évaluation SME: **6.7/10** - Prototype solide, 4 blockers à corriger avant production.

## Priority 1: BLOCKERS (Must Fix)

### B1. Mauvais modèle d'embedding
- **Fichier**: `src/embedding.rs:20`
- **Problème**: `MultilingualE5Small` est un modèle multilingue généraliste, pas adapté au contenu technique/code
- **Impact**: 20-30% moins de recall sur requêtes techniques ("raft consensus", "zero-copy serialization")
- **Solution**: Migrer vers un modèle code-aware:
  - `CodeBERT` (768-dim) - entraîné sur code + documentation
  - `StarEncoder` (768-dim) - conçu pour code search
  - `text-embedding-3-small` (1536-dim) - bonne performance sur code
- [ ] Évaluer les modèles candidats
- [ ] Implémenter la migration (re-vectorisation nécessaire)
- [ ] Benchmark avant/après sur queries techniques

### ~~B2. Foreign keys non activées~~ ✅ DONE
- **Fichier**: `src/db.rs:82`
- [x] `PRAGMA foreign_keys = ON;` ajouté

### ~~B3. Signal handler unsafe~~ ✅ DONE
- **Fichier**: `src/main.rs:200-218`
- [x] Migré de `ctrlc` vers `tokio::signal::ctrl_c()`
- [x] Ajouté `busy_timeout(5s)` pour éviter deadlock
- [x] Exit code 130 (SIGINT) au lieu de 0

### ~~B4. GraphQL cost ignoré~~ ✅ DONE
- **Fichier**: `src/github.rs:992, 1206-1234`
- [x] Ajouté `rateLimit { cost remaining resetAt }` dans la query
- [x] Parse cost/remaining et affiche dans les logs
- [x] Warning si < 500 points restants

---

## Priority 2: HIGH (Should Fix)

### ~~H1. Transactions trop granulaires~~ ✅ DONE
- **Fichier**: `src/db.rs:1116-1123`
- [x] Utilise `prepare_cached()` pour les inserts batch

### H2. Follower discovery incomplet → **WONTFIX**
- Owners sans repos publics n'ont probablement pas de followers intéressants

### H3. Blocking sleep en contexte async → **WONTFIX**
- `embed_missing()` est sync, le sleep est intentionnel (commentaire ajouté)

### H4. Pas de field weighting dans embeddings
- **Fichier**: `src/embedding.rs:100-130`
- **Problème**: Tous les champs ont le même poids, README noie le nom
- **Solution**: Répéter le nom 3x, ajouter marqueurs sémantiques
- [ ] Refactorer `build_embedding_text()` avec weighting
- [ ] Re-vectoriser après changement

### ~~H5. Cache SQLite trop petit~~ ✅ DONE
- **Fichier**: `src/db.rs:86`
- [x] `cache_size = -307200` (300MB)

### H6. Secondary rate limit non géré → **WONTFIX**
- Le backoff existant gère déjà ce cas de façon acceptable

---

## Priority 3: MEDIUM (Nice to Have)

### ~~M1. Composite indexes manquants~~ ✅ DONE
- **Fichier**: `src/db.rs:108-110`
- [x] `idx_repos_gone_embedded` et `idx_repos_gone_id` ajoutés

### M2. Query timeout sur HTTP server
- **Fichier**: `src/http.rs`
- [ ] Ajouter `tokio::time::timeout` sur l'exécution des queries
- [ ] Limiter taille des queries (max 10KB)

### M3. Proxy health scoring
- **Fichier**: `src/proxy.rs`
- **Problème**: Rotation agressive sur timeouts transitoires
- [ ] Tracker success rate par proxy
- [ ] Clear sticky proxy seulement après 5+ échecs consécutifs

### M4. Builder pattern pour GitHubClient
- **Fichier**: `src/github.rs:124-136`
- **Problème**: Trop de paramètres booléens ("boolean trap")
- [ ] Implémenter `GitHubClient::builder().token().debug().build()`

### M5. Connection pooling pour HTTP server
- **Fichier**: `src/http.rs:216`
- [ ] Utiliser `r2d2_sqlite` pour pool de connexions

### M6. Query expansion déterministe
- **Fichier**: `src/search.rs:61-129`
- **Problème**: Dépendance externe à Claude CLI, non-déterministe
- [ ] Implémenter expansion locale avec synonymes techniques
- [ ] Cacher les expansions dans SQLite

---

## Métriques Actuelles

| Métrique | Valeur |
|----------|--------|
| Repos identifiés | 5,090,700 |
| Avec métadonnées | 1,065,937 |
| Avec embeddings | 723,604 |
| Owners distincts | 553,378 |
| Owners à explorer | 1,685,915 |
| Owners sans followers | 42,750 |

---

## Scores SME

| Expert | Score | Verdict |
|--------|-------|---------|
| Search/IR | 5.5/10 | ❌ Modèle d'embedding est un blocker |
| Data Engineer | 7/10 | ⚠️ OK pour <10M repos |
| Rust Engineer | 8/10 | ⚠️ OK pour outils internes |
| GitHub API | 6.25/10 | ❌ Pas prêt pour crawling à grande échelle |
| **Moyenne** | **6.7/10** | **Prototype solide, 4 blockers** |

---

## Références

- `src/embedding.rs` - Génération d'embeddings
- `src/db.rs` - Couche base de données SQLite
- `src/github.rs` - Client API GitHub (REST + GraphQL)
- `src/discovery.rs` - Pipeline de découverte owners/repos
- `src/search.rs` - Recherche sémantique (RRF)
- `src/http.rs` - SQL Explorer web UI
- `src/proxy.rs` - Gestion des proxies
