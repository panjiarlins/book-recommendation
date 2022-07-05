"""
# Sistem Rekomendasi: Rekomendasi Buku Berdasarkan Data Buku Yang Telah Dibaca Oleh Warga di Negara Kanada
<hr>

### *Oleh: [Panji Arlin Saputra](https://www.dicoding.com/users/panjiarlins)*
### *Proyek Akhir: Machine Learning Terapan Dicoding*
<hr>

## **Pendahuluan**
Pada proyek ini, topik yang dibahas adalah mengenai rekomendasi buku berdasarkan data buku yang telah dibaca oleh warga di negara Kanada. Proyek ini dibuat untuk proyek akhir - Machine Learning Terapan Dicoding.

# **1. Mengimpor pustaka/modul python yang dibutuhkan**
"""

# Impor pustaka/ modul
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 
from zipfile import ZipFile
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
import matplotlib.pyplot as plt

"""# **2. Mempersiapkan Dataset**

## **2.1 Menyiapkan kredensial akun Kaggle**
"""

# Membuat folder .kaggle di dalam folder root
!rm -rf ~/.kaggle && mkdir ~/.kaggle/

# Menyalin berkas kaggle.json pada direktori aktif saat ini ke folder .kaggle
!mv kaggle.json ~/.kaggle/kaggle.json
!chmod 600 ~/.kaggle/kaggle.json

"""## **2.2 Mengunduh dan Menyiapkan Dataset**

Informasi Dataset:

| Jenis | Keterangan |
| ----- | ----- |
| Sumber | [Kaggle Dataset - Book-Crossing: User review ratings](https://www.kaggle.com/ruchi798/bookcrossing-dataset) |
| Kategori | Seni dan Hiburan, Komunitas Online, Sastra |
| Jenis dan Ukuran Berkas | CSV (600 MB) |
"""

# Mengunduh dataset menggunakan Kaggle CLI
!kaggle datasets download ruchi798/bookcrossing-dataset

# Mengekstrak berkas zip ke direktori aktif saat ini
!unzip /content/bookcrossing-dataset.zip

"""# **3. Pemahaman Data (_Data Understanding_)**

## **3.1 Memuat Data pada sebuah Dataframe menggunakan pandas**
"""

# Untuk memuat himpunan data
path = '/content/Books Data with Category Language and Summary/Preprocessed_data.csv'
df = pd.read_csv(path)

# Menampilkan sample data pada dataset
df.sample(5)

"""## **3.2 Uraian variabel pada dataset**"""

# Memuat informasi dataframe
df.info()

# Memuat deskripsi kolom pada dataframe
df.describe().round(2)

"""## **3.3 Menangani _missing value_**"""

# Menghitung jumlah data yang kosong pada setiap kolom
df.isna().sum()

# Menghapus data yang memiliki nilai kosong
df = df.dropna()

"""# **4. Persiapan Data (Data Preparation)**

## **4.1 Mengatasi masalah data yang tidak diperlukan dengan menghapus kolom tersebut**
"""

# Menghapus kolom yang tidak diperlukan
df.drop(['Unnamed: 0', 'location', 'img_s', 'img_m', 'img_l', 'Summary', 'city', 'state'], axis=1, inplace=True)
df.sample(5)

"""## **4.2 Pembersihan data pada setiap kolom**

### **4.2.1 Kolom country**
"""

# Menghitung sebaran nilai pada variabel country
df['country'].value_counts()

# Memfilter semua pengguna yang berasal dari negara Kanada
df = df.loc[df['country'].str.contains('canada')]
df.sample(5)

"""### **4.2.2 Kolom rating**"""

# Menghitung sebaran jumlah nilai pada variabel rating
count = df['rating'].value_counts()
percent = 100 * df['rating'].value_counts(normalize=True)
print(pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)}))
count.plot(kind='bar', title='rating');

# Memfilter semua rating agar nilai rating yang diambil berkisar di angka (1 - 10)
df = df.loc[df['rating'] != 0]

# Mengubah rating menjadi nilai float
df['rating'] = df['rating'].values.astype(np.float32)

# Menghitung lagi sebaran jumlah nilai pada variabel rating
count = df['rating'].value_counts()
percent = 100 * df['rating'].value_counts(normalize=True)
print(pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)}))
count.plot(kind='bar', title='rating');

"""### **4.2.3 Kolom Language**"""

# Menampilkan nilai yang ada pada variabel Language
df['Language'].unique()

# Menghapus semua data pada variable Language yang memiliki nilai yang aneh
df = df[df['Language'] != '9']

# Menampilkan lagi nilai yang ada pada variabel Language
df['Language'].unique()

"""### **4.2.4 Kolom Category**"""

# Menampilkan nilai yang ada pada variabel Category
df['Category'].unique()[:50]

# Menghapus semua data pada variable Category yang memiliki nilai yang aneh
df = df[df['Category'].str.startswith('[') & df['Category'].str.endswith(']')]

"""## **4.3 Menampilkan informasi dataset**"""

# Memastikan tidak ada data duplikasi
df.duplicated().sum()

# Mengecek total baris dan kolom dari dataset
df.shape

# Menampilkan informasi data
df.info()

print('Jumlah buku:', len(df.isbn.unique()))
print('Jumlah pengguna:',len(df.user_id.unique()))
print('Jumlah rating', len(df))

"""## **4.4 _Encoding_ variabel user_id dan isbn**"""

# Mengubah user_id menjadi list tanpa nilai yang sama
user_id = df['user_id'].unique().tolist()
print('list user_id: ', user_id)
 
# Melakukan encoding user_id
user_to_user_encoded = {x: i for i, x in enumerate(user_id)}
print('encoding user_id : ', user_to_user_encoded)
 
# Melakukan proses encoding angka ke ke user_id
user_encoded_to_user = {i: x for i, x in enumerate(user_id)}
print('encoding angka ke user_id: ', user_encoded_to_user)

# Mengubah isbn menjadi list tanpa nilai yang sama
book_isbn = df['isbn'].unique().tolist()
 
# Melakukan proses encoding isbn
isbn_to_isbn_encoded = {x: i for i, x in enumerate(book_isbn)}
print('encoded isbn:', isbn_to_isbn_encoded)
 
# Melakukan proses encoding angka ke isbn
isbn_encoded_to_isbn = {i: x for i, x in enumerate(book_isbn)}
print('encoded angka ke isbn:', isbn_encoded_to_isbn)

# Mapping normalized user_id ke dataset
df['user_encoded'] = df['user_id'].map(user_to_user_encoded)
 
# Mapping encoded isbn ke dataset
df['isbn_encoded'] = df['isbn'].map(isbn_to_isbn_encoded)

df

"""## **4.5 Menormalisasi data rating dan melakukan pembagian data pada dataset**"""

# Nilai minimum rating
min_rating = min(df['rating'])
 
# Nilai maksimal rating
max_rating = max(df['rating'])

# Mengacak dataset
df = df.sample(frac=1, random_state=42)

# Membuat variabel x untuk mencocokkan data user dan isbn menjadi satu value
x = df[['user_encoded', 'isbn_encoded']].values
 
# Membuat variabel y untuk menyimpan hasil normalisasi data rating
y = df['rating'].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
 
# Membagi menjadi 80% data train dan 20% data validasi
train_indices = int(0.8 * df.shape[0])
x_train, x_val, y_train, y_val = (
    x[:train_indices],
    x[train_indices:],
    y[:train_indices],
    y[train_indices:]
)

"""## **4.6 Membuat dataset buku**"""

# Menghapus duplikasi pada data buku
preparation = df
preparation = preparation.drop_duplicates('isbn')
preparation

# Menginsialisasikkan variabel buku
book_isbn = preparation['isbn'].tolist()
book_title = preparation['book_title'].tolist()
book_author = preparation['book_author'].tolist()
book_year_of_publication = preparation['year_of_publication'].tolist()
book_publisher = preparation['publisher'].tolist()
book_language = preparation['Language'].tolist()
book_category = preparation['Category'].tolist()

# Membuat dataframe buku
books = pd.DataFrame({
    'isbn': book_isbn,
    'title': book_title,
    'author': book_author,
    'year_of_publication': book_year_of_publication,
    'publisher': book_publisher,
    'language': book_language,
    'category': book_category
})
data = books
books

"""# **5. Pembuatan Model**

## **5.1 _Content Based Filtering_**
"""

# Inisialisasi TfidfVectorizer
tfidf = TfidfVectorizer()

# Melakukan perhitungan idf pada data category
tfidf.fit(data['category'])
print(len(tfidf.get_feature_names()))

# Mapping array dari fitur index integer ke fitur nama
tfidf.get_feature_names()

# Melakukan fit lalu ditransformasikan ke bentuk matrix
tfidf_matrix = tfidf.fit_transform(data['category'])

# Melihat ukuran matrix tfidf
tfidf_matrix.shape

# Mengubah vektor tf-idf dalam bentuk matriks dengan fungsi todense()
tfidf_matrix.todense()

# Membuat dataframe untuk melihat tf-idf matrix
# Kolom diisi dengan category
# Baris diisi dengan judul buku

pd.DataFrame(
    tfidf_matrix.todense(),
    columns=tfidf.get_feature_names(),
    index=data.title
).sample(10)

# Menghitung cosine similarity pada matrix tf-idf
cosine_sim = cosine_similarity(tfidf_matrix) 
cosine_sim

# Membuat dataframe dari variabel cosine_sim dengan baris dan kolom berupa judul buku
cosine_sim_df = pd.DataFrame(cosine_sim, index=data['title'], columns=data['title'])
print('Shape:', cosine_sim_df.shape)
 
# Melihat similarity matrix pada setiap buku
cosine_sim_df.sample(5, axis=1).sample(10, axis=0)

def book_recommendations(title, similarity_data=cosine_sim_df, items=data[['title', 'category']], k=5):
    """
    Rekomendasi buku berdasarkan kemiripan dataframe
 
    Parameter:
    ---
    title : tipe data string (str)
            Judul buku (index kemiripan dataframe)
    similarity_data : tipe data pd.DataFrame (object)
                      Kesamaan dataframe, simetrik, dengan buku sebagai 
                      indeks dan kolom
    items : tipe data pd.DataFrame (object)
            Mengandung judul dan fitur lainnya yang digunakan untuk mendefinisikan kemiripan
    k : tipe data integer (int)
        Banyaknya jumlah rekomendasi yang diberikan
    ---
 
 
    Pada index ini, mengambil k dengan nilai similarity terbesar 
    pada index matrix yang diberikan (i).
    """
 
 
    # Mengambil data dengan menggunakan argpartition untuk melakukan partisi secara tidak langsung sepanjang sumbu yang diberikan
    # Dataframe diubah menjadi numpy
    # Range(start, stop, step)
    index = similarity_data.loc[:,title].to_numpy().argpartition(
        range(-1, -k, -1))
    
    # Mengambil data dengan similarity terbesar dari index yang ada
    closest = similarity_data.columns[index[-1:-(k+2):-1]]
    
    # Drop title agar judul buku yang dicari tidak muncul dalam daftar rekomendasi
    closest = closest.drop(title, errors='ignore')
 
    return pd.DataFrame(closest).merge(items).head(k)

"""## **5.2 _Collaborative Filtering_**"""

class RecommenderNet(tf.keras.Model):
 
  # Insialisasi fungsi
  def __init__(self, num_users, num_isbn, embedding_size, **kwargs):
    super(RecommenderNet, self).__init__(**kwargs)
    self.num_users = num_users
    self.num_isbn = num_isbn
    self.embedding_size = embedding_size
    self.user_embedding = layers.Embedding( # layer embedding user
        num_users,
        embedding_size,
        embeddings_initializer = 'he_normal',
        embeddings_regularizer = keras.regularizers.l2(1e-6)
    )
    self.user_bias = layers.Embedding(num_users, 1) # layer embedding user bias
    self.isbn_embedding = layers.Embedding( # layer embeddings isbn
        num_isbn,
        embedding_size,
        embeddings_initializer = 'he_normal',
        embeddings_regularizer = keras.regularizers.l2(1e-6)
    )
    self.isbn_bias = layers.Embedding(num_isbn, 1) # layer embedding isbn bias
 
  def call(self, inputs):
    user_vector = self.user_embedding(inputs[:,0]) # memanggil layer embedding 1
    user_bias = self.user_bias(inputs[:, 0]) # memanggil layer embedding 2
    isbn_vector = self.isbn_embedding(inputs[:, 1]) # memanggil layer embedding 3
    isbn_bias = self.isbn_bias(inputs[:, 1]) # memanggil layer embedding 4
 
    dot_user_isbn = tf.tensordot(user_vector, isbn_vector, 2) 
 
    x = dot_user_isbn + user_bias + isbn_bias
    
    return tf.nn.sigmoid(x) # activation sigmoid

# Mendapatkan jumlah pengguna
num_users = len(user_to_user_encoded)
 
# Mendapatkan jumlah buku
num_isbn = len(isbn_encoded_to_isbn)

model = RecommenderNet(num_users, num_isbn, 50) # inisialisasi model
 
# model compile
model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = keras.optimizers.Adam(learning_rate=0.001),
    metrics=[tf.keras.metrics.MeanSquaredError()]
)

# Memulai training
 
history = model.fit(
    x = x_train,
    y = y_train,
    batch_size = 8,
    epochs = 100,
    validation_data = (x_val, y_val)
)

"""# **6. Evaluasi Model**

## **6.1 Content-Based Filtering**

### **6.1.1 Menampilkan rekomendasi buku berdasarkan satu sampel buku**
"""

# Menampilkan hasil rekomedasi buku
sample_book_title = 'Flower Painting in Watercolor'
sample_book_category = '(Art)'
sample_book_recommendation = book_recommendations(sample_book_title, k=100)
sample_book_recommendation

"""### **6.1.2 Mengukur metrik evaluasi**"""

total_recommendation = len(sample_book_recommendation)
total_relevant_recommendation = len(sample_book_recommendation[sample_book_recommendation['category'].str.contains('Art')])
print('Sample book:', sample_book_title, sample_book_category)
print('Total recommendation:', total_recommendation)
print('Total recommendation that are relevant:', total_relevant_recommendation)

# Mengukur precision metric
print('Precision Metric:', total_relevant_recommendation/total_recommendation)

"""## **6.2 Collaborative Filtering**

### **6.2.1 Memvisualisasikan hasil pengukuran metrik evaluasi model**
"""

plt.plot(history.history['mean_squared_error'])
plt.plot(history.history['val_mean_squared_error'])
plt.title('model_metrics')
plt.ylabel('mean_squared_error')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

"""### **6.2.2 Menampilkan rekomendasi buku dari salah satu sampel pengguna**"""

books_df = books
df_uji = df
 
# Mengambil sample user
user_id = df_uji.user_id.sample(1).iloc[0]
books_read_by_user = df_uji[df_uji.user_id == user_id]

books_not_read = books_df[~books_df['isbn'].isin(books_read_by_user.isbn.values)]['isbn'] 
books_not_read = list(
    set(books_not_read)
    .intersection(set(isbn_to_isbn_encoded.keys()))
)

books_not_read = [[isbn_to_isbn_encoded.get(x)] for x in books_not_read]
user_encoder = user_to_user_encoded.get(user_id)
user_books_array = np.hstack(
    ([[user_encoder]] * len(books_not_read), books_not_read)
)

# Menampilkan Top-N rekomendasi buku
ratings = model.predict(user_books_array).flatten()
 
top_ratings_indices = ratings.argsort()[-10:][::-1]
recommended_books_ids = [
    isbn_encoded_to_isbn.get(books_not_read[x][0]) for x in top_ratings_indices
]
 
print('Showing recommendations for users: {}'.format(user_id))
print('===' * 9)
print('Books with high ratings from user')
print('----' * 8)
 
top_books_user = (
    books_read_by_user.sort_values(
        by = 'rating',
        ascending=False
    )
    .head(5)
    .isbn.values
)
 
books_df_rows = books_df[books_df['isbn'].isin(top_books_user)]
for row in books_df_rows.itertuples():
    print(row.title, ':', row.category)
 
print('----' * 8)
print('Top 10 books recommendation')
print('----' * 8)
 
recommended_books = books_df[books_df['isbn'].isin(recommended_books_ids)]
for row in recommended_books.itertuples():
    print(row.title, ':', row.category)