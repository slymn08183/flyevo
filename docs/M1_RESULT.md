# M1 sonucu — tüm-beyin modeli evrim sorusu için yeterli mi?

**Karar: NO-GO** (2026-09-19). Tüm-beyin LIF modeli, mantar gövdesi (mushroom body) ile ilgili
sorular için kullanılamaz. Plan §8'deki yola göre devam: evrim deneyi L1 (rate) modelinde yapılır,
"sonuç tüm-beyin bağlamında da tutuyor" iddiasından şimdilik vazgeçilir.

Koşu: `scripts/pretest_shiu.py configs/pretest/m1_sechellia.yaml` → `runs/pretest/m1_sechellia/`.
Karar kuralı koşudan **önce** config'e yazıldı.

## Ne yaptık, neden

Evrim deneyine aylar harcamadan önce tek bir şeyi bilmek istiyorduk: *elimizdeki beyin modeli,
evrimin gerçekte yaptığı türden bir değişikliğe doğru tepki veriyor mu?* Vermiyorsa o modelin
üstünde evrim koşturmanın anlamı yok.

Sözlük (bir kez):
- **Glomerulus:** burundaki aynı tip koku algılayıcılarının beyinde toplandığı düğüm. Her
  glomerulus kabaca "bir koku kanalı". ~50 tane var (DL2d, DM2 … bunların adları).
- **ORN:** burundaki koku algılayıcı nöron. **PN (projeksiyon nöronu):** glomerulustan bilgiyi
  beynin üst merkezlerine taşıyan kablo.
- **Kenyon hücresi (KC):** mantar gövdesinin ana hücresi, ~5000 tane (iki yarıkürede toplam).
  Her koku bunların küçük, kendine özgü bir alt kümesini (~%5-10) yakar. Yazılımcı gözüyle:
  kokuyu seyrek bir hash'e çeviren katman. Koku hafızası bu seyrek kod üzerine kurulur.
- **APL:** Kenyon hücrelerini frenleyen tek dev nöron; seyrekliği o sağlar.
- **MBON:** mantar gövdesinin çıkış nöronları (96 tane); "yaklaş / kaç" kararına giden sinyal.

Gerçek veri (Ellis ve ark. 2024, doi:10.1038/s41467-024-48839-4; tablo:
`src/flyevo/data/comparative/sechellia_vs_mel.yaml`): noni meyvesine uzmanlaşmış *D. sechellia*'da
PN → Kenyon hücresi bağlantıları melanogaster'a göre şöyle kaymış:

| Koku kanalı | Ne algılar | Sechellia'da | Nasıl |
|---|---|---|---|
| DL2d | hekzanoik asit (noni kokusu) | bağlantı payı %1,02 → %3,19 (≈3×) | PN'ler daha çok dal/uç (bouton) yapmış |
| DP1l | asitler | %0,44 → %3,19 (≈7×) | aynı |
| DC3 | farnesol (maya/narenciye) | %4,50 → %1,37 | PN sayısı 3 → 2 |
| VM5v | fermente meyve alkolleri | %3,36 → %1,82 | PN sayısı 3 → 2 |

Deney: FlyWire melanogaster beynini alıp bu dört değişikliği **elle** uyguladık ("sechellia'laştırılmış"
beyin): DL2d ve DP1l PN'lerine aynı oranda yeni Kenyon hücresi ortağı ekledik, DC3 ve VM5v'den
birer PN sildik. Sonra iki beyne de aynı 7 "kokuyu" verdik (bir koku = o glomerulusun bütün
ORN'lerini 100 Hz'de ateşletmek): 4'ü değişen kanallar, 3'ü kontrol (DM2, DA1, VA2 — mantar
gövdesinde değişmediği bildirilenler). Her koşul 5 deneme, iki beyinde aynı rastgele tohumlar.

Beklenen (karar kuralı): değişen kokularda yanıt veren Kenyon hücresi sayısı **ölçülen yönde**
ve gürültüden belirgin biçimde oynamalı (DL2d, DP1l ↑; DC3, VM5v ↓), MBON çıkışı da kaymalı;
kontrol kokularında **hiçbir şey değişmemeli**. 4 kokudan en az 3'ü geçer + kontroller yerinde
durursa GO.

## Ne çıktı

| Koku | Tür | Beklenen | Yanıt veren KC (normal beyin) | (sechellia'laştırılmış) | Geçti mi |
|---|---|---|---|---|---|
| DL2d | değişen | ↑ | 3862 / 5177 | 4583 | evet (ama aşağıya bak) |
| DP1l | değişen | ↑ | 3862 | 4588 | evet (ama aşağıya bak) |
| DC3 | değişen | ↓ | 3863 | 4572 | **hayır** (ters yöne gitti) |
| VM5v | değişen | ↓ | 3868 | 4572 | **hayır** (ters yöne gitti) |
| DM2 | kontrol | – | 3863 | 4577 | **hayır** (değişmemeliydi) |
| DA1 | kontrol | – | 3865 | 4576 | **hayır** |
| VA2 | kontrol | – | 3866 | 4578 | **hayır** |

Tablonun söylediği üç şey:

1. **Model kokuları ayırt etmiyor.** Normal beyinde hangi kokuyu verirsek verelim Kenyon
   hücrelerinin ~%75'i yanıyor (gerçekte %5-10 olmalı) ve **hep aynı hücreler**: farklı kokuların
   yaktığı kümelerin örtüşmesi 0,998 (1 = birebir aynı). MBON çıkışı da kokudan bağımsız
   (DL2d ile DA1 kokusunun MBON vektörleri arasındaki korelasyon 1,0000). Yani bu modelin mantar
   gövdesi koku bilgisi taşımıyor; seyrek hash yerine "bir şey koktu" diyen tek bitlik bir alarm.
2. **Değişiklik her kokuyu aynı şekilde etkiledi.** Sechellia'laştırınca 7 kokunun 7'sinde de
   sayı ~3860 → ~4580 oldu — azalması gerekenler de, hiç değişmemesi gereken kontroller de.
   Eklediğimiz bağlantılar kokuya özgü bir fark yaratmadı, sadece zaten doymuş sisteme biraz daha
   yakıt koydu. İki "geçti" bu yüzden anlamsız.
3. **Doygunluk anında oluşuyor.** İlk 100 ms'de bile tablo aynı (örtüşme 0,996), yani "başta
   seyrek, sonra taşıyor" durumu da yok; erken pencereye bakmak kurtarmıyor.

## Neden böyle oluyor

Bağlantı verisine baktık — bir Kenyon hücresine gelen ortalama girdi (sinaps sayısı, + uyarıcı,
− engelleyici):

| Kaynak | Kenyon hücresi başına |
|---|---|
| **diğer Kenyon hücreleri** | **+73** |
| PN'ler (asıl koku girdisi) | +63 |
| dopamin nöronları | +12 |
| APL (fren) | −19 |

Model her sinapsı "nörotransmitter tahmini × sinaps sayısı" diye tek kalıba sokuyor. Kenyon
hücreleri birbirine çok bağlı ve model bunların hepsini düz uyarı sayıyor → kendi kendini besleyen
bir döngü (pozitif geri besleme). Tek fren olan APL ise modelde sıradan bir nöron gibi ateşliyor
ve refrakter süresi yüzünden ~430 Hz'de tavana vuruyor; gerçekte APL ateşlemeyen, Kenyon
aktivitesiyle orantılı frenleyen özel bir hücre. Frensiz pozitif geri besleme = doygunluk.

Bu bizim port'umuzun hatası değil: aynı simülatör, modelin doğrulandığı devrede (şeker → beslenme
motor nöronu) referans sonuçlarla nöron nöron uyuşuyor (r = 0,95). Sorun modelin kendisinin mantar
gövdesi için fazla kaba olması. Shiu ve ark. da modeli beslenme ve kaşınma devrelerinde doğruladı,
koku hafızasında değil.

## Dürüstlük notları

- DP1l için ×7,25 oranı bu bireyde fazla agresif kaçtı (PN başına ~2000 yeni ortak; bu PN FlyWire'da
  zaten alışılmadık çok Kenyon hücresine bağlı). Daha yumuşak bir uygulama kararı değiştirmezdi:
  normal beyin daha hiçbir şey değiştirmeden doymuş ve kokudan bağımsız.
- Deneme sayısı 5 (pilottan sonra, hesap süresi için 10'dan düşürüldü; karar ölçütü değişmedi).
  Etki büyüklükleri yüzlerce olduğu için deneme sayısı sonucu etkilemiyor.
- "Erken pencere" ölçümü karar kuralında yoktu, keşif amaçlı eklendi; o da aynı şeyi söylüyor.

## Bundan sonra

Plan §8 bu durumu öngörmüştü: evrim L1 (rate) modelinde koşar. L1'de Kenyon seyrekliği tasarım
gereği var (APL normalizasyonu, ~%5 hedef) ve birim testle garanti altında. Tüm-beyin simülatörü
çöpe gitmiyor: doğrulanmış, GPU'da çalışan bir araç ve 3D vitrin; ileride mantar gövdesi düzeltilmiş
bir sürümle B2 bağlam kontrolü yeniden denenebilir (seçenek olarak: Kenyon↔Kenyon uyarısını
kapatıp APL'yi orantılı fren yapmak — ama bu ölçülmemiş parametre ayarı demek, ayrı ve açıkça
"keşif" etiketli bir iş).
