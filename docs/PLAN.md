# flyevo — Plan 1 Altyapı Planı

> **Tek cümle:** *Drosophila melanogaster* koku devresinin (antennal lobe → mushroom body → MBON) gelişimsel parametrelerini ekolojik bir fitness altında in-silico evrimleştirip, *D. sechellia*'da gerçekte gözlenen değişiklikleri **mekanizma düzeyinde** yeniden üretip üretmediğini test etmek.
>
> **Bilimsel iddia (dikkatli hali):** "Güçlü yapay seçilim altında, gerçek türlerde gözlenen devre değişiklikleriyle aynı mekanizmalara ulaşılıyor mu?" — "Evrimi yeniden oynattık" değil.

Kararlar verilmiş, gerekçeleri kısa.

---

## 0. Karar özeti

| Konu | Karar | Gerekçe |
|---|---|---|
| Dil | **Python 3.13** (`>=3.13,<3.15`; 3.14 CI'da "izinli-başarısız") | Tüm çekirdek bağımlılıkların **binary wheel** verdiği en yeni sürüm — bkz. §2.1. 3.14'ü tutan tek şey JAX değil, ekosistemin kenarları |
| Sayısal çekirdek | **JAX** (+ Equinox opsiyonel) | Batch'lenebilir sim, `vmap` ile popülasyon paralelliği, autodiff karşılaştırması bedava |
| Paket yönetimi | **uv** | Hızlı, lockfile, Docker'da tekrarlanabilir |
| Konfigürasyon | **Hydra + OmegaConf** | Composable config grupları; her koşu çözümlenmiş config'i yazar |
| Evrim | **evosax** (CMA-ES, OpenAI-ES, SimpleGA) + **QDax** (MAP-Elites/CMA-ME) + **pymoo** (NSGA-II) | Tekerleği icat etme; üçü de olgun |
| Connectome verisi | **hemibrain v1.2.1** (birincil) + **MaleCNS v1.0** (ikincil), neuprint üzerinden | İkisi de **CC BY 4.0** → ticari kapı açık. FlyWire (CC BY-NC) sadece ön test için |
| Veri erişimi | **neuprint-python + navis** | Standart, bakımlı |
| Ön test modeli | **Shiu et al. 2024** (Brian2, MIT kod, FlyWire v783) | Tek "tüm beyin" referansı; çekirdek pakete **girmez** |
| 3D görüntüleme | **Rerun** (`rerun-sdk`) — web viewer | Python'dan log, tarayıcıda izle, zaman çubuğu; remote'ta çalışır |
| Deney takibi | **MLflow** (Docker içinde, yerel dosya deposu) | Yerel, ücretsiz, aktif (3.16.x, Eyl 2026). **Aim elendi:** son sürüm May 2025, wheel'leri 3.12'de bitiyor — projeyi eski Python'a zincirleyen tek bağımlılık oydu |
| Konteyner | Docker, tek GPU imajı (`python:3.13-slim` + `jax[cuda12]`) | Windows'ta WSL2 + Docker Desktop; remote'ta aynı imaj |
| Lisans | **Apache-2.0** | Patent hibesi; CC BY veriyle uyumlu |
| Test | pytest + hypothesis; CI'da 2 dakikalık "golden" evrim koşusu | Regresyon yakalamak için |

**Yazılması gereken bilimsel kod** (kütüphanelerde yok): (1) genotip→fenotip üreteci, (2) ekolojik fitness, (3) mekanizma-düzeyi karşılaştırma metrikleri, (4) Rerun logger'ları. Tahmini 2–4k satır. Geri kalan her şey hazır.

---

## 1. Bilimsel tasarım (kodun neyi uyguladığı)

### 1.1 Model kademesi

| Kademe | Ne | Nerede kullanılır |
|---|---|---|
| **L0** | Statik graf (neuprint'ten) | Hücre-tipi gruplaması, mimari kısıtlar |
| **L1** | Rate/lineer-nonlineer model: ORN→PN (divisive normalization) → KC (seyrek rastgele genişletme + APL WTA) → MBON (dopamin kapılı plastisite) | **Evrim döngüsünün içi.** JAX, `vmap` ile popülasyon paralel |
| **L2** | LIF nokta nöron (JAX, kendi batch'lenmiş implementasyonumuz) | En iyi genotiplerin doğrulaması (M5) |
| **Shiu** | Tüm beyin LIF — Shiu 2024 denklemleri, **L2 JAX simülatörüyle** koşar (Brian2 değil: CPU'ya bağlı, GPU sürümü GPL) | **Ön test** (M1) ve B2 bağlam kontrolü. Simülatör çekirdekte ve veriden bağımsız; FlyWire verisi (CC BY-NC) ve yükleyicisi yalnızca `scripts/` altında |
| **L3+** | Jaxley (biyofiziksel) | Gelecek; plana dahil değil |

L1'in gerekçesi: 2024–2025 karşılaştırmalı connectomics verisinin dersi "yakın türler arasında topoloji değil parametre değişiyor". Parametre-evrimi için rate modeli yeterli ve 1000× ucuz.

### 1.2 Genotip (evrimleşen şey)

Hücre-tipi düzeyinde, **sürekli** parametre vektörü (~200–300 boyut). Sinaps-sinaps değil.

| Grup | Boyut | Anlam | Biyolojik karşılık |
|---|---|---|---|
| `pn_count[g]` | ~50 (glomerulus) | Glomerulus başına PN sayısı (tam sayıya yuvarlanır) | Nat Commun 2024 "mekanizma 1" |
| `bouton_count[g]` | ~50 | PN başına MB'ye giden bouton sayısı | Nat Commun 2024 "mekanizma 2" |
| `ei_ratio[g]` | ~50 | Glomerulus içi uyarıcı/engelleyici LN ağırlık oranı | Prieto-Godino 2025 |
| `apl_gain` | 1 | APL geri besleme kazancı (KC seyrekliği) | — |
| `kc_threshold` | 1 | KC eşiği | — |
| `orn_gain[g]` | ~50 | Reseptör duyarlılığı (opsiyonel; Or22a genişlemesi için) | Auer 2020 |

**Evrimleşmeyen:** MBON okuma ağırlıkları (bunlar *öğrenilir*, plastisite kuralıyla, her bireyin yaşamı içinde). Mimari (kim kime bağlanır) sabit — hemibrain'den.

### 1.3 Gelişim (genotip → fenotip)

`develop/generator.py`: genotipten **stokastik** bir bağlantı örneği üretir — PN→KC bağlantıları rastgele, ama olasılıklar `pn_count` ve `bouton_count` ile yanlı. Gerçek gelişim gibi gürültülü; fitness **birkaç gelişimsel çekiliş üzerinden ortalanır**. Bu, fenotipin genotipe deterministik bağlı olmadığı gerçeğini korur ve "aynı genotip, farklı bireyler" varyansını ölçmeyi sağlar.

### 1.4 Ortam ve fitness

`env/`: koku paneli + niş ağırlıkları. Aynı kod, farklı niş config'i:

- `niche=melanogaster`: genelci; meyve/maya kokuları yüksek değer, noni nötr
- `niche=simulans`: genelci, hafif farklı panel
- `niche=sechellia`: noni kokuları (hexanoic acid, methyl hexanoate…) yüksek değer, bazı genelci kokular düşük/toksik

Bir bireyin "yaşamı": deneme dizisi → koku sunumu → MBON çıkışı → yaklaş/kaçın kararı → ödül/ceza → plastisite güncellemesi. **Fitness** = niş-ağırlıklı doğru karar oranı − **maliyet terimi** (toplam PN + bouton sayısı × birim enerji). Maliyet terimi şart: Mery & Kawecki'nin gösterdiği takas olmadan model her şeyi büyütür.

Fitness'ın çoklu proxy'si olmalı (doğru karar, öğrenme hızı, ayırt etme keskinliği) — tek proxy'ye aşırı uyum riski.

### 1.5 Evrim algoritmaları ve ne için kullanıldıkları

| Algoritma | Kütüphane | Rol |
|---|---|---|
| **CMA-ES** (Sep-CMA-ES boyut >300 ise) | evosax | Ana optimizer, ~300 sürekli parametre için standart |
| **OpenAI-ES / PGPE** | evosax | Büyük popülasyon, GPU'da `vmap` ile ölçek; CMA-ES ile karşılaştırma |
| **SimpleGA** (mutasyon + seçilim, popülasyon dinamiği) | evosax | Biyolojiye daha yakın; **null model** için gerekli |
| **Nötr sürüklenme (neutral drift)** | kendi sarmalayıcımız (fitness = sabit) | **Null model.** "Rastgele yürüyüş aynı değişiklikleri üretir mi?" |
| **MAP-Elites / CMA-ME** | QDax | Çözüm *uzayının* haritası: alternatif çözümler, "doğa neden diğerlerini seçmedi" |
| **NSGA-II** | pymoo | Fitness vs maliyet Pareto cephesi |
| **Gradyan (autodiff)** | JAX | Karşılaştırma: ES aynı optimuma mı gidiyor? Damıtılabilirlik sorusuna köprü |

evosax işaret kuralına dikkat: **minimize eder** — fitness'ı negatifle. (`evosax>=0.2` API: `es.ask / es.tell`, `es.default_params`, `es.init`.)

### 1.6 "Yakınsadı" ölçütleri (mekanizma düzeyi)

`analysis/mechanism.py`:

1. **Yön uyumu:** glomerulus başına ΔPN ve Δbouton işaretleri, sechellia gerçek verisiyle uyuşuyor mu? (binom testi)
2. **Mekanizma sınıflandırması:** her glomerulusta değişiklik PN sayısıyla mı, bouton sayısıyla mı geldi? Nat Commun 2024'ün ayrımıyla karşılaştır.
3. **Etki büyüklüğü, null'a karşı:** nötr sürüklenme dağılımında aynı uyumun olasılığı (p-değeri + etki büyüklüğü).
4. **Tekrarlanabilirlik:** 10 tohumda (hakem için 30) aynı mekanizmaya ulaşma oranı.
5. **Alternatif çözümler:** QD haritasında eşit-fitness'lı, farklı-mekanizmalı bölgeler var mı? Varsa maliyet/robustluk ekseninde neden elendikleri.

Üç ölçüt düzeyi (parametre / mekanizma / davranış) **baştan** raporlanır; sonuca göre seçilmez.

### 1.7 İleri tahmin (projenin asıl hedefi)

Sechellia deneyi bir **kalibrasyon**dur, son durak değil. Mantık: bilinen bir geçişi (mel → sechellia) evrim operatörü mekanizma düzeyinde yeniden üretiyorsa, aynı operatör **verisi henüz olmayan** bir nişe koşturulur ve çıktı bir tahmin olarak dondurulur.

- **Fitness hiçbir aşamada "hedef türe benzerlik" değildir.** Fitness ekolojiktir (niş + maliyet); gerçek tür verisi yalnızca sınavdır. Aksi döngüsel olur ve ileri tahmini anlamsız kılar.
- **Kapı:** M3 kalibrasyonu `docs/PREREG.md`'deki ölçütleri geçmeden ileri tahmin koşulmaz. Geçemezse bulgu "operatör yetersiz" olarak raporlanır; tahmin üretilmez.
- **Ara sınav (varsa veriyle):** sechellia'ya ayarlanmış operatör, ikinci bir bilinen karşılaştırmada (ör. erecta E/I, simulans ≈ değişim yok negatif kontrolü) **yeniden ayar yapılmadan** sınanır. Tek noktaya kalibre edilmiş bir operatörle ileri gitmek zayıf iddiadır.
- **Hedef nişler:** *D. suzukii* (olgun/taze meyve uzmanı), kaktüs uzmanları (*D. mojavensis*). Her biri yeni bir `configs/env/<tür>.yaml` + koku paneli; kod değişmez.
- **Çıktı tek bir beyin değildir:** tohumlar üzerinden glomerulus başına ΔPN / Δbouton / ΔE-I **dağılımı**, mekanizma sınıflandırması, tekrarlanabilirlik oranı, QD haritasındaki alternatif çözümler.
- **Yanlışlanabilirlik:** tahminler `docs/PREDICTIONS.md`'ye yazılır, tarihlenir, git etiketi + DOI (Zenodo/OSF) ile dondurulur — ilgili türün EM/karşılaştırmalı verisi çıkmadan **önce**. Sonradan değiştirilmez; veri gelince aynı §1.6 metrikleriyle puanlanır.
- **Varsayım (risk):** hedef türün reseptör ayarı melanogaster'a yakın kabul edilir (koku paneli DoOR benzeri mel verisinden). Bilinen reseptör farkları varsa `orn_gain` başlangıcına işlenir, belirtilir.

---

## 2. Teknoloji yığını

```
Python 3.13 · uv · JAX (güncel sürüm, CUDA 12 wheels, Linux/WSL2) · Equinox (opsiyonel)
evosax ≥0.3 · QDax ≥0.5 · pymoo
neuprint-python · navis · fafbseg-py (sadece ön test) · polars · pyarrow
hydra-core · omegaconf · typer (CLI)
rerun-sdk (3D + zaman serisi, web viewer) · matplotlib (statik figür)
mlflow (deney takibi, Docker servisi, yerel dosya deposu)
pytest · hypothesis · ruff · mypy · pre-commit
Docker (cpu / cuda hedefleri) · docker compose profilleri · devcontainer
Opsiyonel extra: brian2 (CeCILL) — sadece [pretest] extra'sında, çekirdeğe girmez
```

### 2.1 Python sürümü kararı (neden 3.13, neden 3.14 değil)

Araştırma yığınında "güncel" = **en yeni CPython** değil, **her bağımlılığın binary wheel gönderdiği en yeni CPython**. Docker'da tek bir kaynak-derleme (Cython/C++) bağımlılığı, sürüm yeniliğinin getirdiği her şeyden pahalıdır. Eylül 2026 itibarıyla durum:

| Paket | Sürüm / tarih | 3.13 | 3.14 | Not |
|---|---|---|---|---|
| JAX | politika: son 45 aydaki tüm Python'lar | ✓ | ✓ | 3.12 ≥ Tem 2027, 3.13 ≥ Tem 2028 destekli |
| Brian2 | 2.10.1, Ara 2025 | ✓ | ✓ (3.14t dahil) | `>=3.12` |
| evosax | 0.3.0, Ağu 2026 | ✓ | ✓ (saf Python) | Apache-2.0 |
| rerun-sdk | 0.38.1, Eyl 2026 | ✓ | ✓ | `cp310-abi3` wheel → her yeni sürümde çalışır |
| MLflow | 3.16.1, Eyl 2026 | ✓ | muhtemel | aktif |
| QDax | 0.5.0, May 2025 | muhtemel (saf Python) | riskli | JAX/flax pinleri eski kalabilir |
| omegaconf/hydra | 2.3.1, Haz 2026 | ✓ (pratikte) | riskli | classifier'lar eski; saf Python |
| flygym | 2.1.0, Haz 2026 | ✓ | ? | `>=3.12` |
| **Aim** | 3.29.1, **May 2025** | **✗ wheel yok** | ✗ | **elendi** |

Karar: `requires-python = ">=3.13,<3.15"`. CI matrisi: 3.13 zorunlu, 3.14 izinli-başarısız. QDax veya omegaconf 3.14'te kırılırsa üst sınır korunur; ikisi de yeşile dönünce 3.14 varsayılan olur. 3.15 (Eki 2026) JAX'e girdikten ~3 ay sonra matrise eklenir. Free-threaded (3.14t) gereksiz: JAX/XLA zaten GIL'i bırakıyor.

`flyevo doctor` yorumlayıcı sürümünü ve bu tablodaki paketlerin kurulu sürümlerini basar.

**Windows notu:** JAX CUDA wheel'leri Linux-only. Her şey **WSL2 (Ubuntu) + Docker Desktop (WSL2 backend) + NVIDIA Container Toolkit** üzerinden koşar. Native Windows'ta JAX-CUDA denemeyin. VS Code "Dev Containers" ile konteynerin içinde geliştirin; repo WSL2 dosya sisteminde (`~/…`) dursun, `/mnt/c/` altında değil (I/O 10× yavaş).

---

## 3. Repo yapısı (monorepo, tek paket)

Kök dizin temiz tutulur: proje dokümanları `docs/` altında, AI-asistan çalışma kuralları `.claude/` altında; ikisi karışmaz.

```
flyevo/
├── README.md                 # tek cümle + GIF + 5 komut
├── LICENSE                   # Apache-2.0
├── pyproject.toml            # uv; extras: [cuda], [pretest], [dev]
├── uv.lock
├── .python-version           # 3.13
├── Makefile                  # make setup / test / lint / data / evolve / viz / docker-*
├── .pre-commit-config.yaml
├── .env.example              # neuprint token şablonu
├── .claude/
│   └── CLAUDE.md             # Claude Code çalışma kuralları (proje dokümanı değil)
├── docs/
│   ├── PLAN.md               # bu dosya
│   ├── M1_RESULT.md          # M1 go/no-go sonucu (M1 bitince yazılır)
│   ├── PREREG.md             # M3 öncesi yazılır, sonra dondurulur
│   └── assets/               # README GIF'i vb.
├── .github/workflows/ci.yml  # CPU: lint + test + 2 dk golden run
├── .devcontainer/devcontainer.json
├── docker/
│   ├── Dockerfile            # multi-stage: base → cpu → cuda
│   └── compose.yaml          # services: flyevo (GPU), mlflow
├── configs/                  # Hydra
│   ├── config.yaml           # defaults listesi
│   ├── model/{rate.yaml, lif.yaml}
│   ├── circuit/{hemibrain_olf.yaml, malecns_olf.yaml}
│   ├── env/{melanogaster.yaml, simulans.yaml, sechellia.yaml, panels/}
│   ├── evo/{cmaes.yaml, openes.yaml, ga.yaml, neutral.yaml, mapelites.yaml, nsga2.yaml}
│   ├── fitness/{default.yaml}
│   └── experiment/{smoke.yaml, pretest.yaml, mel_stability.yaml, sech_main.yaml, qd_map.yaml}
├── src/flyevo/
│   ├── __init__.py
│   ├── cli.py                # `flyevo doctor|data|evolve|analyze|viz`
│   ├── data/                 # neuprint çekme, önbellek, checksum; comparative/ tabloları
│   │   ├── neuprint_fetch.py
│   │   ├── cache.py
│   │   └── comparative/      # sechellia/simulans/mel per-glomerulus PN & bouton (YAML + kaynak DOI)
│   ├── circuit/              # graf → hücre-tipi grupları → JAX dizileri
│   ├── develop/              # genotip → fenotip üreteci (stokastik)
│   ├── models/               # rate.py, lif.py, plasticity.py, normalization.py
│   ├── env/                  # odor_panel.py, niche.py, trial.py
│   ├── fitness/              # objectives.py, cost.py
│   ├── evo/                  # evosax_runner.py, qdax_runner.py, neutral.py, pymoo_runner.py
│   ├── analysis/             # mechanism.py, stats.py, reports.py
│   ├── viz/                  # rerun_log.py (3D + zaman), figures.py
│   └── motifs/               # saf, bağımsız motifler (fly-LSH, APL-WTA, divisive norm) — ileride "flykit" olarak ayrılabilir
├── scripts/
│   ├── pretest_shiu.py       # M1: Shiu modelinde sechellia perturbasyonları
│   └── export_rrd.py
├── tests/
│   ├── unit/                 # üreteç determinizmi, KC seyrekliği, normalizasyon
│   ├── property/             # hypothesis: genotip sınırları → geçerli fenotip
│   └── golden/               # 2 dk CPU evrimi, referans metrik ± tolerans
├── notebooks/                # keşif; kaynak gerçeği değil
├── data/                     # gitignored; önbellek
└── runs/                     # gitignored; Hydra çıktıları, .rrd, mlflow deposu
```

**Kural:** `src/flyevo` içinde çizim yok (yalnızca `viz/`), notebook'a bağımlılık yok, global state yok. Her deney CLI + config ile koşar; notebook sadece `runs/` okur. Modül dosyaları ilgili kilometre taşında, testiyle birlikte eklenir — boş stub dosya açılmaz.

---

## 4. Konfigürasyon (Hydra)

`configs/config.yaml`:
```yaml
defaults:
  - model: rate
  - circuit: hemibrain_olf
  - env: melanogaster
  - evo: cmaes
  - fitness: default
  - experiment: smoke
  - _self_

seed: 0
n_dev_draws: 4          # gelişimsel çekiliş sayısı (fitness ortalaması)
device: auto            # cpu | cuda | auto
run_dir: runs/${now:%Y-%m-%d}/${now:%H-%M-%S}_${experiment.name}
log:
  mlflow: true
  mlflow_uri: file:runs/mlflow     # ya da http://mlflow:5000 (compose)
  rerun: true
  rerun_every: 10       # her N nesilde 3D snapshot
```

`configs/evo/cmaes.yaml`:
```yaml
name: cmaes
algo: CMA_ES
population_size: 128
generations: 1000
sigma_init: 0.1
```

`configs/env/sechellia.yaml`:
```yaml
name: sechellia
panel: noni_specialist       # env/panels/*.yaml
niche_weights:
  noni: 3.0
  generic_fruit: 0.5
  toxic: -2.0
trials_per_life: 200
```

`configs/experiment/sech_main.yaml`:
```yaml
name: sech_main
seeds: [0,1,2,3,4,5,6,7,8,9]
start_from: runs/mel_stability/best_genotype.npz   # melanogaster'da stabilize edilmiş başlangıç
compare_to: comparative/sechellia_vs_mel.yaml
```

Komut örnekleri:
```bash
flyevo evolve experiment=sech_main env=sechellia evo=cmaes seed=3
flyevo evolve experiment=sech_main evo=neutral            # null model
flyevo evolve experiment=qd_map evo=mapelites
flyevo analyze runs/2026-.../                               # mekanizma metrikleri + figürler
flyevo viz runs/2026-.../ --serve                           # Rerun web viewer
```

Her koşu `run_dir`'e yazar: çözümlenmiş config, git SHA, veri checksum'ları, `genotypes.npz`, `metrics.parquet`, `viz.rrd`.

---

## 5. Docker ve remote

`docker/Dockerfile` tek hedeflidir: `python:3.13-slim` + uv + `--extra cuda`. `nvidia/cuda` tabanı bilerek yok — `jax[cuda12]` wheel'leri CUDA/cuDNN'i kendi getirir, yalnızca host sürücüsü gerekir. `docker/compose.yaml` servisleri: `flyevo` (GPU), `mlflow` (UI `:5000`, SQLite backend `runs/mlflow/` altında). Güncel hali için dosyaların kendisine bakın.

Kullanım:
```bash
docker compose -f docker/compose.yaml up -d mlflow
docker compose -f docker/compose.yaml run --rm flyevo evolve experiment=smoke
```

**Remote GPU (RunPod / Vast / Lambda):**
1. CI imajı GHCR'a push eder: `ghcr.io/<user>/flyevo:cuda`.
2. Remote'ta aynı imajı çek, `runs/`'ı volume olarak bağla, `flyevo evolve …` koş.
3. Sonuçları `rsync`/`rclone` ile yerele çek; `.rrd` dosyalarını yerelde Rerun ile aç. Canlı izlemek için SSH tüneli: `ssh -L 9090:localhost:9090 …`.
4. Tohumlar ayrı işler olarak paralel koşar (`seed=0..9` → 10 konteyner ya da tek konteynerde `vmap`).

**Windows yerel:** `wsl --install -d Ubuntu`, Docker Desktop → Settings → Resources → WSL integration açık; `nvidia-smi` WSL içinde görünmeli. `make docker-cuda` çalışıyorsa hazır.

---

## 6. 3D görüntüleme (Rerun)

Neden Rerun: Python'dan `rr.log(...)` ile 3D geometri + zaman serisi + skalarlar; tarayıcıda izlenir; **zaman ekseni olarak nesil kullanılabilir** → "evrimin videosu". `.rrd` dosyası kaydedilir, remote'tan indirilip yerelde oynatılır.

`viz/rerun_log.py` sorumlulukları:
- **Statik sahne:** hemibrain'den SWC iskeletleri (navis) → `rr.LineStrips3D`; AL glomerulusları, MB lobları mesh/point olarak. Bir kez, `timeless`.
- **Nesil ekseni:** `rr.set_time_sequence("generation", g)`; glomerulus renkleri `pn_count[g]`'ye göre (renk = genotip), boyut = bouton.
- **Yaşam ekseni:** seçilen bireyin denemeleri: `rr.set_time_seconds("trial_time", t)`; KC aktivasyonu nokta bulutu olarak, MBON çıkışı skalar.
- **Skalarlar:** en iyi/ortalama fitness, maliyet, seyreklik, sechellia uyum skoru — `rr.Scalars`.
- **Karşılaştırma paneli:** gerçek sechellia Δ vs evrimleşen Δ, glomerulus başına çubuk.

Rerun API sürümler arasında değişti (`serve_web` → `serve_grpc` + web viewer). **Kurulu sürümün dokümanına göre yaz**, sürümü `pyproject`'te sabitle.

"Oyun gibi" hissi: nesil çubuğunu sürükleyince beyin renk değiştirir; bireye tıklayınca yaşamı oynar. Bu Rerun'un yerleşik davranışı, ekstra kod yok.

---

## 7. Veri

| Kaynak | Ne | Lisans | Nasıl |
|---|---|---|---|
| hemibrain v1.2.1 | AL, MB, LH nöronları + sinapslar; PN/KC/MBON/APL tipleri | CC BY 4.0 | `neuprint-python`, dataset `hemibrain:v1.2.1`, token gerekli (ücretsiz) |
| MaleCNS v1.0 | Aynısı, ikinci örnek | CC BY 4.0 | dataset `male-cns:v1.0`, ya da Feather dosyaları |
| FlyWire v783 | Sadece Shiu ön testi | **CC BY-NC 4.0** | `scripts/pretest_shiu.py` içinde; çekirdeğe girmez; README'de belirtilir |
| Karşılaştırmalı tablolar | sechellia/simulans/mel per-glomerulus PN & bouton; erecta E/I | makalelerin ek tabloları | `src/flyevo/data/comparative/*.yaml`, her satırda DOI |

`flyevo data fetch` → önbelleğe çeker, SHA256 yazar; `flyevo doctor` checksum'ları doğrular. Ham veri repoya girmez; türetilmiş küçük tablolar (< 1 MB) girer.

---

## 8. Kilometre taşları ve kapılar

| # | Hafta | Çıktı | Kapı |
|---|---|---|---|
| **M0 "Hello fly"** | 1 | Repo iskeleti; `make setup/test/lint` yeşil; GPU Docker imajı build; `flyevo doctor` ✓; hemibrain koku alt grafı önbellekte; Rerun'da nöronlar 3D | CI yeşil, GIF README'de |
| **M1 Ön test** | 2 | (a) `models/lif.py`: Shiu modelinin veriden bağımsız JAX portu (GPU, event-driven). (b) `scripts/`: FlyWire v783 yükleyici + **port doğrulaması** — Shiu'nun yayınlanmış sonucu (şeker GRN → MN9) yeniden üretilmeli. (c) `viz/rerun_log.py`: tüm beyin 3D nokta bulutu + ateşleme zaman ekseni. (d) `scripts/pretest_shiu.py`: sechellia'nın bilinen değişikliklerini elle uygula; MBON/davranış çıkışı kayıyor mu? | **Go/No-go.** Kaymıyorsa: L1 modeline geç ama "tüm beyin bağlamı" iddiasından vazgeç; makale "model yeterliliği" makalesine döner |
| **M2 L1 model** | 3–5 | `develop/`, `models/rate.py`, `plasticity.py`; birim + özellik testleri; bilinen nitel sonuçlar: KC seyrekliği ~%5, APL normalizasyonu, tek-deneme öğrenme | Golden test yeşil |
| **M3 Ana deney** | 5–7 | `env/`, `fitness/`; `mel_stability` (melanogaster nişinde evrim → uzaklaşmamalı); `sech_main` 10 tohum; `neutral` null; mekanizma metrikleri | Ön kayıt tarzı bir `docs/PREREG.md` M3 başında yazılır ve değişmez |
| **M4 Harita** | 7–9 | QD haritası, NSGA-II Pareto, gradyan karşılaştırması; analiz notebook'u; figürler | — |
| **M5 Doğrulama + yazı** | 9–12 | En iyi genotiplerin L2 LIF'te doğrulaması; Rerun demo videosu; preprint taslağı | Preprint |
| **M6 İleri tahmin** | 12+ | §1.7: ara sınav (yeniden ayarsız ikinci karşılaştırma); hedef niş config'leri (*suzukii*, kaktüs uzmanı); 10–30 tohum + null; `docs/PREDICTIONS.md` dondurulur (etiket + DOI) | **Yalnızca M3 kalibrasyonu PREREG ölçütlerini geçtiyse.** Tahmin dondurulduktan sonra değişmez |

M1 **flykit'ten bile önce** — sonucu planı değiştirebilir.

---

## 9. Test ve CI

- **Birim:** üreteç aynı seed → aynı fenotip; genotip sınırları → geçerli fenotip; normalizasyon ölçek-değişmez.
- **Özellik (hypothesis):** rastgele genotip → KC seyrekliği [%2, %15] aralığında; fitness sonlu.
- **Golden:** `experiment=smoke` (pop 16, 20 nesil, CPU, < 2 dk) → referans metrik ± tolerans. Sayısal kayma yakalar.
- **CI:** GitHub Actions, CPU; Python matrisi **3.13 (zorunlu) + 3.14 (`continue-on-error`)**; `ruff`, `mypy`, `pytest`, golden. CUDA imajı ayrı iş, sadece build.
- **Pre-commit:** ruff format/lint, büyük dosya engeli (> 5 MB), notebook çıktısı temizleme (`nbstripout`).

---

## 10. "Startup'ta patlamayalım" listesi

1. `git init` → `.gitignore` (data/, runs/, *.rrd, .venv, __pycache__, .hydra) **ilk commit'te**.
2. `.python-version` dosyası `3.13`; `pyproject.toml`: `requires-python = ">=3.13,<3.15"`; extras: `cuda = ["jax[cuda12]"]`, `pretest = ["brian2", "fafbseg"]`, `dev = [...]`.
3. `uv lock` → lock dosyası commit'lenir. Her bağımlılık sürüm sabitli.
4. `flyevo doctor`: Python sürümü, JAX backend, GPU görünürlüğü, veri checksum, Rerun sürümü, MLflow erişimi, §2.1 tablosundaki paket sürümleri → tek komutla "ortam sağlıklı mı".
5. `make setup` = `uv sync --all-extras` + `pre-commit install` + `flyevo data fetch --minimal`.
6. İlk 3 gün hiç bilim yok: iskelet, Docker, CI, doctor, 3D "hello". Sonra M1.
7. `notebooks/` sadece keşif; her bulgu `src/` + test'e taşınmadan "sonuç" sayılmaz.
8. Ham veri ve `runs/` repoya asla girmez; küçük türetilmiş tablolar girer.
9. Her koşu git SHA + config + checksum yazar; SHA kirliyse (`-dirty`) koşu uyarı verir.

---

## 11. Riskler ve karşılıklar

| Risk | Olasılık | Karşılık |
|---|---|---|
| M1 ön testi düşer (Shiu modeli perturbasyona duyarsız) | orta | L1'e geç, iddiayı daralt; "model yeterliliği" bulgusu da yayınlanabilir |
| Genotip belirsiz (farklı parametre kombinasyonları aynı davranış) | yüksek | QD haritası tam bunu ölçer; mekanizma sınıflandırmasını **çoklu çözüm** olarak raporla |
| Tek proxy'ye aşırı uyum | orta | 3 fitness proxy + maliyet terimi; duyarlılık analizi |
| Karşılaştırmalı tablo eksik/belirsiz | orta | Makale yazarlarına e-posta; eksikse o glomerulusları analizden çıkar, belirt |
| Rerun/evosax API kayması | düşük | sürüm sabitleme; `doctor` sürüm kontrolü |
| 3.14'te QDax/omegaconf kırılması | orta | `<3.15` üst sınırı; 3.14 CI'da izinli-başarısız; yeşile dönünce varsayılanı yükselt |
| MLflow/SQLite tek kullanıcı kilidi (paralel tohumlar) | düşük | Tohumlar ayrı run olarak yazar; çakışırsa `file:` store yerine compose'daki server'a geç |
| Tek noktaya kalibrasyon (yalnızca sechellia) ileri tahmini zayıflatır | yüksek | §1.7 ara sınav: ikinci bilinen karşılaştırma yeniden ayarsız; negatif kontrol (simulans); iddia "dağılım + mekanizma" ile sınırlı |
| Hedef türün reseptör ayarı bilinmiyor | orta | Mel paneli varsayımı açıkça yazılır; bilinen farklar `orn_gain`'e işlenir; duyarlılık analizi |
| Windows/WSL2 GPU sorunları | orta | İlk gün `nvidia-smi` WSL'de doğrulanır; olmazsa remote'a geç, yerelde CPU |
| Hesap | düşük | A/B2 senaryoları: $0–400 |

---

## 12. Sonraki adımlar

1. ~~Repo iskeleti (§3), `pyproject.toml` extras ile, Apache-2.0.~~ ✅
2. ~~`Makefile`, `Dockerfile`, `compose.yaml`, `devcontainer.json`, CI.~~ ✅ (imajlar henüz build edilmedi)
3. `flyevo doctor` ve `flyevo data fetch` (hemibrain koku alt grafı: PN, KC, MBON, APL, LN tipleri; neuprint sorguları önbellekli).
4. `viz/rerun_log.py`: iskeletleri 3D çiz, `.rrd` kaydet. README'ye GIF.
5. `scripts/pretest_shiu.py` (M1) — Shiu repo'sunu `[pretest]` extra'sıyla kur, perturbasyon listesi `src/flyevo/data/comparative/sechellia_vs_mel.yaml`'dan.
6. M2'ye geç.

Her adım kendi PR'ı; her PR'da test.

---

## Kararlar günlüğü

- **2026-09-18:** `PLAN.md` → `docs/PLAN.md`, `CLAUDE.md` → `.claude/CLAUDE.md`, `PREREG.md`/`M1_RESULT.md` → `docs/`. Gerekçe: public repo'da kök dizin temiz kalsın; proje dokümanı ile AI-asistan kuralları ayrışsın.
- **2026-09-18:** Python 3.12 → **3.13** (`>=3.13,<3.15`), Aim → **MLflow**, evosax ≥0.3. Gerekçe §2.1: Aim'in wheel'leri 3.12'de bitiyordu ve projeyi eski Python'a zincirliyordu.
- **2026-09-18:** `uv.lock` üretildi (jax 0.11.2, qdax 0.5.0, evosax 0.3.1, mlflow 3.16.1, rerun-sdk 0.38.1); 3.13'te test/lint/mypy/doctor yeşil, QDax güncel JAX ile import oluyor. `[tool.uv] constraint-dependencies` ile `pykdtree>=1.4` (macOS hariç): navis'in macOS'a özel `<=1.3.12` sınırı, wheel'siz sürümü tüm platformlara yayıp Docker'da C derlemesine yol açıyordu (§2.1 kuralı).
- **2026-09-18:** CPU Docker hedefi ve compose profilleri kaldırıldı; tek GPU imajı, ince taban. Ayrı WSL Ubuntu distro'su gereksiz bulundu ve kaldırıldı (Docker Desktop GPU'yu kendi WSL2 backend'iyle geçiriyor). CI şimdilik devre dışı (pipeline yok). Gerekçe: koşular yalnızca GPU'da yapılacak, disk kısıtlı.
- **2026-09-18:** **M6 İleri tahmin** eklendi (§1.7). Sechellia deneyi kalibrasyon olarak yeniden çerçevelendi; asıl hedef, kalibre edilmiş operatörü verisi olmayan nişlere koşturup tarihli, dondurulmuş, yanlışlanabilir tahmin üretmek.
- **2026-09-18:** M1 yeniden tanımlandı: Brian2 yerine Shiu modelinin JAX portu (`models/lif.py`, L2 ile aynı kod), önüne port-doğrulama adımı, ardından tüm-beyin 3D sahne. Tüm-beyin için MaleCNS'e geçiş **yapılmıyor**: ek nöronlar (VNC, erkeğe özgü devreler) koku sorusuna katkı sağlamıyor; tek kazanç lisans, o da tüm beyin yalnızca `scripts/` içinde kullanıldığı sürece fark yaratmıyor. MaleCNS koku alt devresi için ikinci örnek rolünde kalır.
- **2026-09-19:** `models/lif.py` tüm-beyin ölçeğinde GTX 1650'de ölçüldü (sentetik ağ: 138 bin nöron, 15M bağlantı, ~200 aktif nöron): yoğun tarama ~150 s / biyolojik saniye → event-driven ~23 s. 30 denemelik 1 s'lik bir Shiu deneyi ≈ 12 dk. Gerçek beyinde aktivite daha yüksek olabilir; ayar düğmesi `edge_budget` / `max_spikes_per_step`. `XLA_PYTHON_CLIENT_PREALLOCATE=false` Dockerfile'a `ENV` olarak da kondu (CLI'dan geçmeyen script'ler 4 GB kartta OOM ile çöküyordu).
- **2026-09-19:** Port doğrulaması, kaba düzey: FlyWire v783 üzerinde Shiu'nun örnek deneyi (20 şeker GRN'i @150 Hz, 30 × 1 s) → **423 aktif nöron, 467.570 spike, MN9 96 ± 6 Hz**; uyarılan GRN'ler ortalama 147,8 Hz. Referans notebook (v630 verisi) "~400 aktif nöron, >400 bin spike" diyor → uyumlu. GTX 1650'de 30 deneme 605 s (deneme başına ~20 s). Nöron-nöron karşılaştırma (referans `sugarR.parquet` ile) henüz yapılmadı.
- **2026-09-19:** Port doğrulaması, nöron düzeyi (referans: Shiu reposundaki hazır sonuç dosyaları, v630 verisi; bizimki v783). 100 Hz: uyarılan GRN'ler 99,0 ↔ 99,3 Hz, aktif nöron 404 ↔ 406, MN9 67,0 ↔ 70,5 Hz, iki tarafta da aktif 340 nöronda r = 0,95. Bizde toplam spike ~%13 fazla — connectome sürüm farkıyla tutarlı ama kanıtlanmadı; birebir karşılaştırma v630 bağlantı dosyasını (86,6 MB) gerektirir. Not: repodaki `sugarR.parquet` notebook'taki 150 Hz ile değil ~200 Hz ile üretilmiş görünüyor (GRN'ler ~197 Hz). Tüm-beyin 3D sahne (`viz.rrd`) ve GIF üretildi; aktivite SEZ'de toplanıyor.
- **2026-09-19:** **M1 = NO-GO** (`docs/M1_RESULT.md`). Tüm-beyin LIF'te mantar gövdesi doymuş ve kokuya kör: her kokuda Kenyon hücrelerinin ~%75'i, hep aynı hücreler (kümeler arası örtüşme 0,998), MBON çıkışı kokudan bağımsız. Sechellia düzenlemeleri 7 kokunun 7'sini de aynı yönde oynattı (kontroller dahil). Sebep: KC→KC girdisi (+73 sinaps/KC, PN girdisinden fazla) düz uyarı sayılıyor, tek spike'lı APL freni yetmiyor. Karar kuralı koşudan önce yazılmıştı. Sonuç §8'deki yola göre: evrim L1 rate modelinde; "tüm-beyin bağlamı" iddiası şimdilik bırakıldı. Gerçek karşılaştırma verisi `src/flyevo/data/comparative/sechellia_vs_mel.yaml`'a işlendi (Ellis ve ark. 2024).
