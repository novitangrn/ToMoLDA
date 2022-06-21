# ToMoLDA
#Projek ‘Klasifikasi Teks dalam NLP untuk Mendeteksi Topik Berita Berbasis Teks’ dibuat untuk mendapatkan sebaran topik  dari dokumen-dokumen yang ada dalam sebuah korpus. Data yang diambilberupa berita berbasis teks yang bisa didapatkan melalui proses scraping web secara real-time, atau bisa juga dengan mengunggah dataset lokal di menu yang telah disediakan. Kemudian dataset yang telah didapatkan diproses dengan menggunakan LDA (Latent Dirichlet Allocation). Metode LDA ini memudahkan dalam mencari atau memunculkan topik-topik tersembunyi saat melakukan pemrosesan serta dapat mendeteksi seberapa proporsionalnya kemunculan sebuah topik-topik tertentu

#Libraries
import requests
from bs4 import BeautifulSoup
import pandas as pd

import nltk
import numpy as np
import re

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

import pyLDAvis
import pyLDAvis.gensim 

from nltk.corpus import stopwords
nltk.download('stopwords')
list_stopwords = stopwords.words('indonesian')
list_stopwords.extend(["lengkap", "shutterstockcom", "newsletter", "senin", "freepikcom", "pexelscom",
                       "jam", "selasa", "rabu", "kamis", "jumat", "sabtu", "minggu", "grafis",
                       "ikuti", "komentar", "tag", "perbincangan", "home", "seleb", "gaya", "freepikcomxb",
                       "wib", "font", "arial", "roboto", "times", "verdana", "ukuran", "bagikan", 
                       "berita", "&amp", "xd", "your", "browser", "does", "not", "support", "the", 
                       "video", "tagxd", "cari", "berdasarkan", "info", "bisnis", "event", "komentar", 
                       "juta", "unit", "menang", "artikel", "persen", "smartphone", "tagar", "shutterstock", 
                       "travel", "home", "komentar", "berita", "judul", "diubah", "topik", "newsletter", "tempo", 
                       "tempoco", "logo", "pixabaycom", "nama", "tempodian", "temposubekti", "foto", "jakarta", 
                       "difabel", "ilustrasi", "januari", "februari", "maret", "april", "mei", "juni", "juli", 
                       "agustus", "september", "oktober", "november", "desember", "istimewa", "dr", "ja",
                       "tag", "bincang", "gaya", "bincang", "seleb", "ukur", "event", "politifact", "salah", "freepik",
                       "lupa", "sandi", "masuk", "google", "facebook", "batal", "laki", "perempuan", "kerja",
                       "memiliki", "akun", "daftar", "konfirmasi", "email", "kirim", "baca", "meni",
                       "link", "aktivasi", "klik", "non", "laput", "koran", "majalah", "pixabay",
                       "tempo", "menerima", "masukan", "alamat", "mereset", "password", "reset",  "update", 
                       "politik", "ekonomi", "metro", "sepak", "bola", "kesehatan", "teknologi", "otmotif",
                       "milik", "terima", "kirim", "dunia", "daerah", "hapus", "sehat", "kota", "kerja", "didik", "open"
                      ])
list_stopwords.append("lengkap")                      
list_stopwords = set(list_stopwords)

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
factory = StemmerFactory()
stemmer = factory.create_stemmer()
import warnings
warnings.filterwarnings("ignore")

# Core Packages
import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter.scrolledtext import *
from tkinter.filedialog import askopenfilename, asksaveasfilename
from tkinter import messagebox 
from datetime import date

# Structure and Layout
window = Tk()
window.title("Topic Modeling GUI")
window.geometry("700x400")
window.config(background='wheat')
style = ttk.Style(window)
style.configure('lefttab.TNotebook', tabposition='wn', background='whitesmoke', font='Helvatica')

tab_control = ttk.Notebook(window,style='lefttab.TNotebook')
tab1 = ttk.Frame(tab_control)
tab2 = ttk.Frame(tab_control)
tab3 = ttk.Frame(tab_control)
tab4 = ttk.Frame(tab_control)

# ADD TABS TO NOTEBOOK
tab_control.add(tab1, text=f'{"Get the Dataset":^30s}')
tab_control.add(tab2, text=f'{"ToMo by File":^30s}')
tab_control.add(tab3, text=f'{"ToMo by Scrape":^30s}')
tab_control.add(tab4, text=f'{"About":^30s}')

# ADD LABEL
label1 = Label(tab1, text= 'Dapatkan Dataset', padx=150, pady=5, font='Arial 15 bold', anchor=N)
label1.grid(column=0, row=0)
label2 = Label(tab2, text= 'ToMo: Topic Modeling with LDA (Latent Dirichlet Allocation) by File Upload')
label3 = Label(tab1, text= 'ToMo: Topic Modeling with LDA (Latent Dirichlet Allocation) by Scrape Web')             
label4 = Label(tab4, text= 'ABOUT', padx=150, pady=5, font='Arial 15 bold', anchor=N)
label4.grid(column=0, row=0)
tab_control.pack(expand=1, fill='both')

#FUNCTIONS---------------------------------------------------------------------------------

def ScrapeWeb():
    tempo = requests.get('https://www.tempo.co')
    beautify = BeautifulSoup(tempo.content)
    terkini = beautify.find('main',{'class':'main-left'})
    berita = terkini.find_all('div',{'class':'card-box ft240 margin-bottom-sm'})
    upperframe = []
    frame = []
    for each in berita:
        title = each.find('h2', {'class':'title'}).text
        link = each.a.get('href')
        res = requests.get(link)
        html_page = res.content
        soup = BeautifulSoup(html_page, 'html.parser')
        text = soup.find_all(text=True)
        output = ''
        blacklist = ['label',
                     'h4', 
                     'ol', 
                     '[document]', 
                     'a', 
                     'h1', 
                     'h2',
                     'noscript', 
                     'header', 
                     'ul', 
                     'html', 
                     'section', 
                     'article', 
                     'em', 
                     'meta', 
                     'title', 
                     'body', 
                     'aside', 
                     'footer', 
                     'div', 
                     'form', 
                     'nav',  
                     'head', 
                     'link', 
                     'strong', 
                     'h6', 
                     'br', 
                     'li', 
                     'h3',
                     'href',
                     'h5', 
                     'input', 
                     'blockquote', 
                     'main', 
                     'script', 
                     'img',
                     'figure',
                     'style',
                     'hr',
                     'kbd',
                     'time',
                     'samp'
                    ]    
        for t in text:
            if t.parent.name not in blacklist:
                output += '{} '.format(t)
        frame.append((link, title, output))
    upperframe.extend(frame)
    global data
    data=pd.DataFrame(frame, columns=['Link','Judul', 'Text'])
    return data

def scrape_web():
    messagebox.showwarning("Warning", "Proses ini akan memakan waktu cukup lama.")
    ScrapeWeb()
    entry.insert(tk.END, data)
    
def upload_file():
    f_types = [('CSV files',"*.csv"),('All',"*.*")]
    file = filedialog.askopenfilename(filetypes=f_types)
    l1.config(text=file) # display the path
    try:
        global korpus
        korpus = pd.read_csv(file) # create DataFrame
        entry.insert(tk.END, korpus) # add to Text widget
    except ValueError:
        messagebox.showerror('Information', 'File is invalid')
    except FileNotFoundError:
        messagebox.showerror('Information', "File not found")

def save_csv():
    today = date.today().strftime('%m%d%Y') # to set the date in the csv filename
    to_csv = data.to_csv('berita_{}.csv'.format(today))
    return to_csv

class KorpusLda():
    def __init__(self, korpus):
        self.df = korpus
        self.df = self.df.drop(columns=['Judul', 'Link'])
        self.corpus_superlist = self.df[['Text']].values.tolist()
        self.corpus = []
        for sublist in self.corpus_superlist:
            for item in sublist:
                self.corpus.append(item)

    def preprocessing(self):
        def preprocess_text(document):
            document = document.lower()
            document = re.sub(r'\W', ' ', str(document))
            document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
            document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)
            document = re.sub(r'\s+', ' ', document)
            document = re.sub(r'^b\s+', '', document)
            document = re.sub('[-+]?[0-9]+', '', document)
            tokens = document.split()
            tokens = [stemmer.stem(word) for word in tokens]
            tokens = [word for word in tokens if len(word)>5]
            tokens = [word for word in tokens if word not in list_stopwords]            
            return tokens
        global processed_data, gensim_corpus, gensim_dictionary
        processed_data = [];
        for doc in self.corpus:
            tokens = preprocess_text(doc)
            processed_data.append(tokens)
        gensim_dictionary = corpora.Dictionary(processed_data)
        gensim_corpus = [gensim_dictionary.doc2bow(token, allow_update=True) for token in processed_data]
        return gensim_corpus, gensim_dictionary

    def modeling(self, gensim_corpus, gensim_dictionary):
        lda_model = gensim.models.ldamodel.LdaModel(corpus=gensim_corpus, num_topics=5, id2word=gensim_dictionary, passes=50)
        return lda_model
    
    def plotting(self, lda_model, gensim_corpus, gensim_dictionary):
        pyLDAvis.enable_notebook()
        vis_data = pyLDAvis.gensim.prepare(lda_model, gensim_corpus, gensim_dictionary)
        return vis_data

    def performance(self, lda_model, gensim_corpus, gensim_dictionary):
        print('\nPerplexity:', lda_model.log_perplexity(gensim_corpus))
        coherence_score_lda = CoherenceModel(model=lda_model, texts=processed_data, dictionary=gensim_dictionary, coherence='c_v')
        coherence_score = coherence_score_lda.get_coherence()
        print('\nCoherence Score:', coherence_score)
        
    def main():
        if __name__ == '__main__':
            #file = ScrapeWeb()
            global lda_proses, lda_model, df
            lda_proses = KorpusLda(korpus)
            gensim_corpus, gensim_dictionary = lda_proses.preprocessing() 
            lda_model = lda_proses.modeling(gensim_corpus, gensim_dictionary)            
            model = lda_model.show_topics(num_topics = 5, num_words = 10)
            df = pd.DataFrame(model)
            df.columns = ["Index Topik", "Kata Pembangun Topik"]
            return df
    
    def test():
        if __name__ == '__main__':
            global performa
            performa = lda_proses.performance(lda_model, gensim_corpus, gensim_dictionary)
            return performa  
        
    def plot():
        if __name__ == '__main__':
            lda_plot = lda_proses.plotting(lda_model, gensim_corpus, gensim_dictionary)
            savevis = pyLDAvis.save_html(lda_plot, 'lda.html')
            return savevis

class ScrapeLda():
    def __init__(self, data):
        self.df = data
        self.df = self.df.drop(columns=['Judul', 'Link'])
        self.corpus_superlist = self.df[['Text']].values.tolist()
        self.corpus = []
        for sublist in self.corpus_superlist:
            for item in sublist:
                self.corpus.append(item)

    def preprocessing(self):
        def preprocess_text(document):
            document = document.lower()
            document = re.sub(r'\W', ' ', str(document))
            document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
            document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)
            document = re.sub(r'\s+', ' ', document, flags=re.I)
            document = re.sub(r'^b\s+', '', document)
            document = re.sub('[-+]?[0-9]+', '', document)
            tokens = document.split()
            tokens = [stemmer.stem(word) for word in tokens]
            tokens = [word for word in tokens if word not in list_stopwords]
            tokens = [word for word in tokens if len(word)  > 3]
            return tokens
        global processed_data, gensim_corpus, gensim_dictionary
        processed_data = [];
        for doc in self.corpus:
            tokens = preprocess_text(doc)
            processed_data.append(tokens)
        gensim_dictionary = corpora.Dictionary(processed_data)
        gensim_corpus = [gensim_dictionary.doc2bow(token, allow_update=True) for token in processed_data]
        return gensim_corpus, gensim_dictionary

    def modeling(self, gensim_corpus, gensim_dictionary):
        lda_model = gensim.models.ldamodel.LdaModel(corpus=gensim_corpus, num_topics=5, id2word=gensim_dictionary, passes=50)
        return lda_model
    
    def plotting(self, lda_model, gensim_corpus, gensim_dictionary):
        pyLDAvis.enable_notebook()
        vis_data = pyLDAvis.gensim.prepare(lda_model, gensim_corpus, gensim_dictionary)
        return vis_data

    def performance(self, lda_model, gensim_corpus, gensim_dictionary):
        print('\nPerplexity:', lda_model.log_perplexity(gensim_corpus))
        coherence_score_lda = CoherenceModel(model=lda_model, texts=processed_data, dictionary=gensim_dictionary, coherence='c_v')
        coherence_score = coherence_score_lda.get_coherence()
        print('\nCoherence Score:', coherence_score)
        
    def main():
        if __name__ == '__main__':
            global lda_proses, lda_model, df
            lda_proses = ScrapeLda(data)
            gensim_corpus, gensim_dictionary = lda_proses.preprocessing() 
            lda_model = lda_proses.modeling(gensim_corpus, gensim_dictionary)            
            model = lda_model.show_topics(num_topics = 5, num_words = 10)
            df = pd.DataFrame(model)
            df.columns = ["Index Topik", "Kata Pembangun Topik"]
            return df
    
    def test():
        if __name__ == '__main__':
            global performa
            performa = lda_proses.performance(lda_model, gensim_corpus, gensim_dictionary)
            return performa  
        
    def plot():
        if __name__ == '__main__':
            lda_plot = lda_proses.plotting(lda_model, gensim_corpus, gensim_dictionary)
            savevis = pyLDAvis.save_html(lda_plot, 'lda.html')
            return savevis
        
def format_topic(ldamodel, corpus, texts): 
    df_topic = pd.DataFrame()
    for i, row_list in enumerate(ldamodel[corpus]):
        row = row_list[0] if ldamodel.per_word_topics else row_list            
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  #=> dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                df_topic = df_topic.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    df_topic.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']
    contents = pd.Series(texts)
    df_topic = pd.concat([df_topic, contents], axis=1)
    return(df_topic)

def data_hasil():
    topic_keywords = format_topic(ldamodel=lda_model, corpus=gensim_corpus, texts=processed_data)
    dominant_topic = topic_keywords.reset_index()
    dominant_topic.columns = ['Index Dokumen', 'Topik Dominan', 'Persentase', 'Kata Kunci', 'Teks']
    return dominant_topic 
        
def korpus_tomo():
    messagebox.showwarning("Warning", "Proses ini akan memakan waktu cukup lama.")
    KorpusLda.main()
    entry2.insert(tk.END, df)
    
def korpus_hasil():
    nilai = data_hasil()
    tab2_display.insert(tk.END, nilai)
    
def korpus_vis():
    KorpusLda.plot()
    
def scrape_tomo():
    messagebox.showwarning("Warning", "Proses ini akan memakan waktu cukup lama.")
    ScrapeLda.main()
    entry3.insert(tk.END, df)
    
def scrape_vis():
    ScrapeLda.plot()

def scrape_hasil():
    nilai = data_hasil()
    tab3_display.insert(tk.END, nilai)

# 1ST TAB
# Buttons
button1=Button(tab1,text="Scrape Web", command=scrape_web, width=12,bg='darkseagreen',fg='#0c1435')
button1.grid(row=2,column=0,padx=5,pady=5)

button2=Button(tab1,text="Simpan File CSV", command=save_csv, width=12,bg='darkseagreen',fg='#0c1435')
button2.grid(row=3,column=0,padx=5,pady=5)

button3=Button(tab1, text="Buka File CSV", command=upload_file, width=12, bg='darkseagreen', fg='#0c1435')
button3.grid(row=4,column=0,padx=5,pady=5)

# Label
l1=Label(tab1,text="Dapatkan file corpus yang akan diproses")
l1.grid(row=1,column=0)

entry=ScrolledText(tab1,height=20)
entry.grid(row=5,column=0,columnspan=5,padx=5,pady=5)


# 2ND TAB
# Buttons
button3=Button(tab2, text="Membangun Topik", command=korpus_tomo, width=20, bg='darkseagreen', fg='#0c1435')
button3.grid(row=2,column=0,padx=5,pady=5)

button4=Button(tab2, text="Sebaran Topik", command=korpus_hasil, width=20, bg='darkseagreen', fg='#0c1435')
button4.grid(row=6,column=0,padx=5,pady=5)

button5=Button(tab2, text="Simpan Visualisasi", command=korpus_vis, width=20, bg='darkseagreen', fg='#0c1435')
button5.grid(row=15,column=0,padx=5,pady=5)

# Label
l2=Label(tab2,text="Temukan topik yang ada di setiap dokumen yang telah diunggah")
l2.grid(row=1,column=0)

entry2=ScrolledText(tab2, height=8)
entry2.grid(row=4,column=0,columnspan=3,padx=5,pady=5)

# Display Screen For Result
tab2_display = Text(tab2, height=20)
tab2_display.grid(row=8, column=0, columnspan=8, padx=5, pady=5)
tab2_display.config(state=NORMAL)

# Display screen for visualization
vis_text = Label(tab2, text='Display otomatis tersimpan pada penyimpanan lokal dengan nama file lda.html.')
vis_text.grid(row=20, column=0)

# 3RD TAB
# Buttons
button6=Button(tab3, text="Membangun Topik", command=scrape_tomo, width=20, bg='darkseagreen', fg='#0c1435')
button6.grid(row=2,column=0,padx=5,pady=5)

button7=Button(tab3, text="Sebaran Topik", command=scrape_hasil, width=20, bg='darkseagreen', fg='#0c1435')
button7.grid(row=6,column=0,padx=5,pady=5)

button8=Button(tab3, text="Simpan Visualisasi", command=scrape_vis, width=20, bg='darkseagreen', fg='#0c1435')
button8.grid(row=15,column=0,padx=5,pady=5)

# Label
l3=Label(tab3,text="Temukan topik yang ada di setiap dokumen")
l3.grid(row=1,column=0)

entry3=ScrolledText(tab3, height=8)
entry3.grid(row=4,column=0,columnspan=3,padx=5,pady=5)

# Display Screen For Result
tab3_display = Text(tab3, height=20)
tab3_display.grid(row=8, column=0, columnspan=8, padx=5, pady=5)
tab3_display.config(state=NORMAL)

# Display screen for visualization
vis_text = Label(tab3, text='Display otomatis tersimpan pada penyimpanan lokal dengan nama file lda.html.')
vis_text.grid(row=20, column=0)

# 3RD TAB
about = Label(tab4,text="Klasifikasi Teks dalam NLP untuk Deteksi Topik Berita Berbasis Teks \n \n DISUSUN OLEH KELOMPOK VI \n Algoritma dan Pemrograman Lanjut C083 \n \n 1. Novita Anggraini - 21083010104 \n 2. Galang Surya Ramadhan - 21083010081 \n 3. Awal Lidya Musaffak - 21083010088 \n 4. Radya Ardi Ninang Pudyastuti - 21083010097 \n5. Mohamad Ibnu Fajar Maulana - 21083010106 \n", padx=5, pady=5)
about.grid(column=0,row=1)
              
window.mainloop()
