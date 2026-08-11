# Geri Yayılım (Backpropagation) Algoritmasının Mantığı ve Mekaniği: Detaylı Kılavuz

## 1. Giriş ve Temel Kavramlar

Yapay zeka ve derin öğrenme dünyasının kalbinde yer alan geri yayılım (backpropagation) algoritması, bir sinir ağının yaptığı hatalardan nasıl ders çıkaracağını belirleyen temel mekanizmadır. Karmaşık matematiksel formüllerin ötesine geçtiğimizde, bu algoritmanın aslında bir ağın parametrelerini sistematik bir şekilde "iyileştirme" süreci olduğunu görürüz. Bir eğitim tasarımcısı olarak yaklaşımımız, bu süreci soyut bir matematik yığını olarak değil, sezgisel bir "hassasiyet ayarı" olarak kavramaktır.

Algoritmanın işleyişini anlamak için öncelikle sinir ağının üç ana bileşenine hakim olmalıyız:

* **Ağırlıklar (Weights):** Nöronlar arasındaki bağlantıların gücünü temsil eder. Bilginin bir katmandan diğerine ne kadar "güçlü" aktarılacağını belirlerler.
* **Önyargılar (Biases):** Bir nöronun aktif hale gelmesi (ateşlenmesi) için aşması gereken eşik değeridir.
* **Maliyet Fonksiyonu (Cost Function):** Ağın tahminlerinin gerçeklikten ne kadar uzak olduğunu ölçen "başarısızlık puanı"dır. Bizim amacımız bu puanı sıfıra yaklaştırmaktır.

Bu temelleri kavradığımızda, ağın bir görüntüyü nasıl işlediğini ve hatalarını nasıl düzelttiğini daha somut bir senaryo üzerinden inceleyebiliriz.

## 2. Örnek Senaryo: El Yazısı Rakam Tanıma (MNIST)

Geri yayılımın gücünü anlamak için endüstri standardı olan MNIST (el yazısı rakam tanıma) veri kümesini kullanıyoruz. Bu senaryoda tasarladığımız mimari şu şekildedir:

* **Giriş Katmanı:** 784 nöron (28x28 piksellik bir görüntünün her bir pikselini temsil eder).
* **Gizli Katmanlar:** Her biri 16 nörondan oluşan iki adet ara katman.
* **Çıkış Katmanı:** 10 nöron (0'dan 9'a kadar olan rakamları temsil eder).

Bu mimari, toplamda yaklaşık 13.000 adet ağırlık ve önyargı (parametre) içerir. Ağ bir görüntüyü aldığında, bu 13.000 parametre üzerinden bir hesaplama yapar ve çıkış katmanındaki nöronlara belirli aktivasyon değerleri atar. Ancak ağın sadece bir tahminde bulunması yeterli değildir; bu tahminin ne kadar hatalı olduğunu ölçmemiz ve bu devasa parametre yığınını doğru yöne yönlendirmemiz gerekir.

## 3. Maliyet Fonksiyonu ve Negatif Gradyan İlişkisi

Ağın ne kadar "kötü" performans gösterdiğini anlamak için maliyet fonksiyonuna başvururuz. Bu süreçte şu adımları izleriz:

1. Ağın mevcut çıktı değerlerini alırız.
2. Bu değerleri, olması gereken ideal çıktılarla karşılaştırırız.
3. Hataların karelerini alarak toplar ve on binlerce örnek üzerinden ortalamayı buluruz.

İşte bu noktada negatif gradyan kavramı devreye girer. Gradyan, sadece bir sayılar listesi değil, 13.000 boyutlu uzayda maliyetin en dik artış yönünü gösteren bir vektördür. Biz maliyeti düşürmek istediğimiz için bu vektörün tersine, yani negatifine yöneliriz. Negatif gradyan, ağdaki her bir parametrenin (ağırlık ve önyargı) maliyeti en hızlı şekilde düşürmek için hangi yönde ve ne kadar "dürtülmesi" gerektiğini bize söyler.

## 4. Hassasiyet Analizi: Gradyan Büyüklüklerinin Anlamı

Gradyan vektöründeki her bir bileşenin büyüklüğü, maliyet fonksiyonunun o belirli parametreye karşı ne kadar "hassas" olduğunu gösterir. Bu hassasiyet, 13.000 boyutlu vektörün hangi eksenlere daha fazla eğildiğini belirler.

> **Hassasiyet Karşılaştırması:**  
> "Hesaplamalarımız sonucunda bir ağırlığın gradyan bileşeni 3.2 çıkarken, bir diğerinin 0.1 çıktığını varsayalım. Bu, maliyet fonksiyonunun ilk ağırlıktaki değişikliklere karşı 32 kat daha hassas olduğu anlamına gelir. Yani, ilk ağırlığı küçük bir miktar değiştirmek, maliyeti düşürmede ikinci ağırlığa göre 32 kat daha büyük bir etki yaratacaktır. Algoritma, 'paranın karşılığını' en iyi veren bu hassas parametrelere öncelik verir."

Bu hassasiyetleri anladığımızda, ağın tek bir eğitim örneği karşısında nasıl "dürtüldüğünü" görebiliriz.

## 5. Tek Bir Örnek Üzerinden Sezgisel Adımlar: Rakam "2" Örneği

Eğitilmemiş bir ağa "2" rakamını gösterdiğimizi düşünelim. Ağın rastgele çıktıları ile hedefimiz arasındaki fark, "dürtme" miktarını belirler. Önemli olan kural şudur: Hedef değerden ne kadar uzaktaysak, o nöronu düzeltme isteğimiz o kadar güçlü (büyük) olur.

| Çıkış Nöronu | Mevcut Aktivasyon | Hedef Değer | Gereken Dürtme Şiddeti |
| :--- | :--- | :--- | :--- |
| **Rakam 1** | 0.8 | 0.0 | Çok Güçlü (Azalmalı) |
| **Rakam 2** | 0.2 | 1.0 | Çok Güçlü (Artmalı) |
| **Rakam 8** | 0.3 | 0.0 | Zayıf (Azalmalı) |

Aktivasyonları doğrudan değiştiremeyiz, ancak bu hedeflere ulaşmak için bir önceki katmandaki ağırlıkları ve önyargıları manipüle edebiliriz.

## 6. Ağırlık ve Önyargıları Dürtme Yolları ve Hebbian Teorisi

Bir nöronun (örneğin "2" nöronu) aktivasyonunu artırmak için üç kaldıraç kullanırız:

1. **Önyargıyı Artırmak:** Nöronun ateşleme eşiğini düşürür.
2. **Ağırlıkları Ayarlamak:** Burada "paranın karşılığı" ilkesi geçerlidir. Önceki katmandaki en parlak (en aktif) nöronlarla olan bağlantıların ağırlığını artırmak, sönük nöronlara göre çok daha büyük bir etki yaratır.
3. **Önceki Katmanı Değiştirmek:** Pozitif ağırlıklı bağlantıların daha aktif, negatiflerin ise daha pasif olmasını "arzulamak".

> **Uzman Görüşü (Hebbian Teorisi):**  
> Bu mekanizma, sinirbilimdeki ünlü *"Birlikte ateşleyen nöronlar, birlikte bağlanır"* (*Neurons that fire together, wire together*) ilkesini anımsatır. Yapay sinir ağlarında da, halihazırda aktif olan bir nöron ile aktif olması istenen bir sonraki nöron arasındaki bağ en çok güçlendirilen bağdır.

Bu yerel isteklerin ağın geneline nasıl yayıldığı, algoritmanın asıl dehasını oluşturur.

## 7. İsteklerin Geriye Doğru Yayılması (Backpropagating Desires)

Geri yayılım ismi, bu "dürtme isteklerinin" çıkıştan girişe doğru bir zincirleme reaksiyon şeklinde iletilmesinden gelir. Tek bir çıkış nöronu değil, 10 çıkış nöronunun tamamı önceki katmandaki nöronlara *"Şu kadar artmalısın"* veya *"Şu kadar azalmalısın"* şeklinde talepler gönderir.

* **İsteklerin Toplanması:** Gizli katmandaki her bir nöron, kendisinden sonra gelen tüm nöronlardan farklı şiddette talepler alır.
* **Ağırlıklı Talep Yönetimi:** Bu talepler, aradaki bağlantı ağırlıklarıyla çarpılarak toplanır.
* **Özyinelemeli (Recursive) Süreç:** Bu toplam talepler, bir önceki katman için yeni "hedef değişimler" haline gelir.

Süreç, katman katman geriye doğru tekrarlanır. Çıkış katmanındaki hata payı, giriş katmanına kadar tüm ağırlık ve önyargılara "ne kadar değişmeleri gerektiği" bilgisi olarak dağıtılır.

## 8. Tüm Örneklerin Birleşimi ve Stokastik Gradyan İnişi (SGD)

Ağın öğrenmesi için on binlerce örneğin taleplerini dengelemek gerekir. Her adımda tüm veri setini hesaplamak (Tam Gradyan İnişi) muazzam bir hesaplama gücü gerektirir ve çok yavaştır. Bu yüzden modern yapay zekada **Stokastik Gradyan İnişi (SGD)** tercih edilir.

| Özellik | Tam Gradyan İnişi | Stokastik Gradyan İnişi (SGD) |
| :--- | :--- | :--- |
| **Veri Kullanımı** | Tüm eğitim seti (Örn: 60.000 veri) | Küçük rastgele gruplar (Mini-batch) |
| **Hız ve Verimlilik** | Çok yavaş ve hantal | Çok hızlı ve verimli |
| **İlerleme Biçimi** | Her adımda en kesin yokuş aşağı yön | Hızlı ama "Sarhoş Adam" gibi tökezleyerek |

> **"Sarhoş Adam" Benzetmesi:**  
> SGD kullanan bir algoritma, her adımda en mükemmel yönü hesaplamak için durup bekleyen dikkatli bir adam yerine; küçük bir veri grubuna bakarak hızlı ama biraz sarsak adımlarla yokuş aşağı koşan bir adama benzer. Bu sarsak adımlar başlangıçta kaotik görünse de, sağladığı devasa hız avantajı sayesinde ağın maliyet fonksiyonundaki yerel minimuma çok daha kısa sürede ulaşmasını sağlar.

**Temel Çıkarım:** Geri yayılım, her bir eğitim örneğinin sesini dinleyip bu seslerin ortalamasını alan demokratik ve sistematik bir hata düzeltme sürecidir. Bu sürekli ayarlama döngüsü, basit bir piksel yığınının zamanla karmaşık kavramları tanıyan bir zekaya dönüşmesini sağlar.
