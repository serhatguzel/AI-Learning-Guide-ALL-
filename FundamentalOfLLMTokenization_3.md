# 🏗️ LLM Dünyasının Görünmez Mimarisi: GPT Tokenizasyon Teknik Raporu

## 📖 1. Giriş: Tokenizasyon - LLM'lerin Görünmez Alfabesi
Büyük Dil Modelleri (LLM), metinleri bizim gibi kelimeler veya harfler bazında değil, **token** adı verilen birimler üzerinden algılar. Tokenizasyon, basit bir metin parçalama işlemi değil; bir modelin dünyayı algılama biçimini belirleyen "atomik" temelidir. 

Karakterler çok küçük ve anlamsız kalırken, tam kelimeler sözlük boyutunu yönetilemez hale getirir. Bu nedenle LLM'ler, metni anlamlı "parçacıklara" (*chunks*) ayırarak orta bir yol bulur.

### 🎯 Etki Analizi: Dil, Matematik ve Mantık Sınırları
Tokenizasyonun tasarımı, modelin başarısını ve operasyonel maliyetini doğrudan belirler. Özellikle Türkçe gibi eklemeli dillerde ve teknik görevlerde şu sorunlar baş gösterir:

* 🇹🇷 **Türkçe Dil Dezavantajı:** Tokenizer eğitim setindeki İngilizce ağırlığı nedeniyle Türkçe kelimeler BPE tarafından tam birleştirilemez. Kelimeler anlamsız ek/harf seviyelerine parçalanır. Bu "genleşmiş" (*stretched*) temsil, aynı metnin İngilizceye göre çok daha fazla token harcamasına (yüksek fatura) ve bağlam penceresinin (*context window*) hızla dolmasına neden olur.
* 🔤 **İmla ve Heceleme Zayıflığı:** Model bir kelimeyi (örneğin "default") tek bir atom olarak gördüğünde, içindeki harfleri tek tek manipüle etmekte (tersten yazma, harf sayma) zorlanır.
* 🔢 **Matematiksel Kararsızlık:** Sayıların (örneğin 127, 677) tokenizer tarafından rastgele parçalara bölünmesi, modelin aritmetik mantık kurmasını engeller.
* 💻 **Kodlama Verimsizliği:** Python gibi boşluk hassasiyeti olan dillerde her boşluğun ayrı bir token olması, modelin mantıksal kapasitesini daraltır.

*Bu verimlilik sorunlarının kökenine inmek için, bu parçalamayı yapan temel mekanizmaya, yani BPE algoritmasına bakmamız gerekir.*

---

## 🌳 2. Byte Pair Encoding (BPE) Algoritması: 'Binary Forest' İnşası

BPE, metni ham bayt seviyesinden (UTF-8) alarak en sık kullanılan bayt çiftlerini yeni bir token olarak birleştirme prensibiyle çalışır.

### Mekanizma ve İkili Orman (Binary Forest)
Karpathy’nin tabiriyle BPE süreci aşağıdan yukarıya bir "Binary Forest" inşasıdır:
1. **Yapraklar (Leaves):** En altta temel 256 UTF-8 bayt değeri bulunur.
2. **Birleşme (Merge):** Algoritma, veri seti içinde yan yana en sık gelen iki birimi bulur ve bunları yeni bir düğüm (*node*) olarak birleştirir.
3. **İç Düğümler (Internal Nodes):** Her birleşme işlemi, hiyerarşide yeni bir üst düğüm oluşturur. Bu süreç, belirlenen sözlük boyutuna ulaşana kadar devam eder.

### ⚖️ Sözlük Boyutu ve 'Sweet Spot' (Altın Denge)
Sözlük boyutu (*Vocabulary Size*), modelin yoğunluğu ve hesaplama maliyeti arasındaki dengeyi belirler.

| Parametre | Sözlük Boyutu Arttıkça | Etkisi |
| :--- | :--- | :--- |
| **Dizi Yoğunluğu** | ⬆️ Artar | Aynı metin daha az token ile ifade edilir, Latent Space verimliliği artar. |
| **Bellek Maliyeti** | ⬆️ Artar | Gömme Tablosu (*Embedding Table*) büyür, daha fazla VRAM gerektirir. |
| **Hesaplama Yükü** | ⬆️ Artar | Çıkış katmanındaki Softmax hesaplama karmaşıklığı ve gecikmesi (*latency*) artar. |
| **Eğitim Kalitesi** | ⚠️ Riskli | Nadir token'lar yeterince veri görmezse (*under-training*), model bu tokenları anlamlandıramaz. |

> 💡 **Analoji:** Eğer sözlükte her nadir kelimeye yer verirseniz, o kelimeyi hayatında sadece bir kez duyan bir çocuk gibi model de o parametreleri düzgün eğitemez.

*Algoritmayı anladığımıza göre, bu süreci endüstride uygulayan devlerin yöntemlerini kıyaslayabiliriz.*

---

## 🥊 3. Kütüphane Karşılaştırması: Tiktoken (OpenAI) vs SentencePiece (Google)

Endüstride iki temel yaklaşım öne çıkar. OpenAI **Tiktoken** ile doğrudan bayt seviyesine odaklanırken, Google **SentencePiece** ile kod noktaları (*code points*) üzerinden ilerler.

### Teknik Farklar ve Optimizasyon
SentencePiece, Tiktoken'dan farklı olarak "Byte Fallback" (Nadir karakterler için bayt seviyesine dönüş) ve "Dummy Prefix" (Metin başına otomatik boşluk ekleme) gibi mekanizmalar kullanır. Ancak en çarpıcı fark kod yönetiminde görülür:

* **GPT-2 (İsraf Dönemi):** Her bir boşluk karakteri (token 220) ayrı işleniyordu. Bu, Python girintilerinin bağlam penceresini gereksiz doldurmasına yol açıyordu.
* **GPT-4 (Yoğun Girdi):** Boşluk grupları (3, 4, 7 boşluk gibi) tek bir token'da birleştirilir. Bu "yoğunlaştırma" sayesinde modelin kodlama kapasitesi ve çıkarım hızı artar.

    // GPT-2 Temsili (Verimsiz - Her boşluk bir atom)
    [space][space][space][space][if][space][x][>][0]

    // GPT-4 Temsili (Yoğunlaştırılmış / Dense Input)
    [4-spaces-combined][if][space][x][>][0]

*Standart token'lar dışında, modelin akışını kontrol eden "özel görevli" belirteçlere ve onların sisteme dahil edilme sürecine odaklanalım.*

---

## 🪚 4. Özel Token'lar ve Model Ameliyatı (Model Surgery)

Bazı token'lar metnin parçası değil, modelin kontrol mekanizmasıdır. `<|endoftext|>` belgenin bittiğini, FIM (*Fill-In-the-Middle*) belirteçleri ise metin tamamlama mantığını yönetir. Mevcut bir modele yeni özel belirteçler ekleme süreci (Model Ameliyatı) şu adımları izler:

1. **Gömme Matrisi (Embedding Matrix) Genişletme:** Sözlüğe yeni bir satır eklenerek yeni token için yer açılır.
2. **Başlatma (Initialization):** Bu yeni satırdaki parametreler başlangıçta rastgele gürültü (*random noise*) olarak atanır; yani token henüz hiçbir anlam taşımaz.
3. **LM Head Çıkış Katmanı Güncellemesi:** Modelin son katmanındaki tahmin birimi sayısı yeni boyuta göre artırılır.
4. **İnce Ayar (Fine-Tuning):** Model eğitilirken gelen gradyan güncellemeleriyle, o rastgele gürültü olan parametreler anlam kazanır *(Örn: "bu token gelirse konuşmayı bitir")*.

*Mükemmel görünen bu mimaride bile, bazen modelin "aklını kaçırmasına" neden olan çok tehlikeli tuzaklar gizlidir.*

---

## 🚩 5. Meşhur Hatalar ve Kritik 'Footgun' Uyarıları

Tokenizasyon, dikkat edilmediğinde modeli "vuran" gizli tuzaklarla (*footguns*) doludur.

| Hata Adı | Neden | Sonuç |
| :--- | :--- | :--- |
| **Solid Gold Magikarp** | Tokenizer eğitim setinde olup model eğitim setinde olmayan tokenlar (Reddit kullanıcı adları vb.). | Gömme matrisindeki bu satırlar hiç eğitilmediği için "ayrılmamış bellek" gibi davranır; model saçmalar veya hakaret eder. |
| **Sondaki Boşluk (Trailing Whitespace)** | Metin sonundaki boşluğun, bir sonraki kelimenin parçası olması gereken boşluğu "çalması". | Model veri dağılımı dışına (*OOD - Out-of-Distribution*) çıkar ve performans dramatik şekilde düşer. |
| **Unstable (Kararsız) Token'lar** | Modelin sadece bütün halinde bildiği bir token'ın (Örn: *default style*) parçalanarak sunulması. | Model bu alt-token dizilimlerini eğitimde hiç görmediği için OOD hatası verir, güvenlik filtrelerini tetikler veya durur. |

*Tüm bu teknik detaylar ve riskler ışığında, bir geliştiricinin izlemesi gereken somut adımları belirleyelim.*

---

## 🏆 6. Geliştiriciler İçin Altın Kurallar ve Özet

Bir LLM mimarı veya geliştiricisi olarak uygulama geliştirirken şu kurallara riayet edilmelidir:

- [x] **`.strip()` Kullanımına Dikkat:** Kullanıcı girdilerindeki boşlukları kontrolsüzce silmek, modelin beklediği (özellikle boşlukla başlayan) token dizilimlerini bozarak OOD sorunlarına yol açabilir.
- [x] **Karakter Ayrıştırma (Spacing) Stratejisi:** Modelin heceleme veya matematik yapması gerekiyorsa, girdiyi karakterlerine ayırarak `(h e l l o)` vermek, modelin "atomik körlüğünü" yenmesini sağlar.
- [x] **Tokenizer-Model Senkronizasyonu:** Eğitimdeki tokenizer konfigürasyonu ile çıkarım (*inference*) anındaki yapılandırma %100 örtüşmelidir.

### 📌 Özet Çıkarımlar
* **Tokenizasyon bir algı motorudur:** LLM dünyayı karakterlerle değil, bu özel alfabeyle görür.
* **BPE bir sıkıştırma sanatıdır:** Dizi yoğunluğu (*density*) ile hesaplama maliyeti arasındaki kritik dengeyi kurar.
* **Görünmez tehlikeler gerçektir:** Solid Gold Magikarp gibi "eğitilmemiş" tokenlar, modelin en zayıf halkasıdır.

> **Sonuç:** Tokenizasyon, LLM geliştirme sürecinde genellikle bir külfet olarak görülse de, doğru anlaşıldığında model performansını optimize eden ve operasyonel hataları minimize eden kritik bir sanattır.
