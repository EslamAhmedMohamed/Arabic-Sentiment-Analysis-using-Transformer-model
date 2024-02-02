!pip install tashaphyne
import gensim
from gensim.models import KeyedVectors
from gensim.models import word2vec

!unzip '/kaggle/working/AraVec'
!rm AraVec
import nltk
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from tashaphyne.stemming import ArabicLightStemmer
import re
import emoji
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
from sklearn.preprocessing import LabelEncoder
nltk.download('stopwords')
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Data preprocessing ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

test=pd.read_csv("/kaggle/input/neural/test_no_label.csv")
df=pd.read_csv("/kaggle/input/neural/train1.csv")
#print(df)
df = df.iloc[:]
# print("print data", df)
# print(len(df))
# Remove duplicated
df.review_description.duplicated().sum()
df.drop(df[df.review_description.duplicated() == True].index, axis=0, inplace=True)


# Remove Punctuation
df.review_description = df.review_description.astype(str)
df.review_description = df.review_description.apply(
    lambda x: re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', x))
df.review_description = df.review_description.apply(lambda x: x.replace('؛', "", ))
# print("print data after pun", df.head())


# Define the function to remove consecutive duplicated Arabic words
def remove_duplicate_arabic_words(text):
    # Tokenize the text into words
    words = text.split()

    # Remove consecutive duplicated words
    unique_words = [words[i] for i in range(len(words)) if i == 0 or words[i] != words[i - 1]]

    # Join the unique words back into a sentence
    modified_text = ' '.join(unique_words)

    return modified_text


df['review_description'] = df['review_description'].apply(remove_duplicate_arabic_words)
# Remove StopWords
stopWords = list(set(stopwords.words("arabic")))  ## To remove duplictes and return to list again

# Some words needed to work with to will remove
for word in ['لا', 'لكن', 'ولكن']:
    stopWords.remove(word)
df.review_description = df.review_description.apply(
    lambda x: " ".join([word for word in x.split() if word not in stopWords]))
# print("print data after removing stop words", df.head())

# Replace Emoji by Text
emojis = {"🙂": "يبتسم", "😂": "يضحك", "💔": "قلب حزين", "🙂": "يبتسم", "❤": "حب", "❤": "حب", "😍": "حب", "😭": "يبكي",
          "😢": "حزن", "😔": "حزن", "♥": "حب", "💜": "حب", "😅": "يضحك", "🙁": "حزين", "💕": "حب", "💙": "حب", "😞": "حزين",
          "😊": "سعادة", "👏": "يصفق", "👌": "احسنت", "😴": "ينام", "😀": "يضحك", "😌": "حزين", "🌹": "وردة", "🙈": "حب",
          "😄": "يضحك", "😐": "محايد", "✌": "منتصر", "✨": "نجمه", "🤔": "تفكير", "😏": "يستهزء", "😒": "يستهزء", "🙄": "ملل",
          "😕": "عصبية", "😃": "يضحك", "🌸": "وردة", "😓": "حزن", "💞": "حب", "💗": "حب", "😑": "منزعج", "💭": "تفكير",
          "😎": "ثقة", "💛": "حب", "😩": "حزين", "💪": "عضلات", "👍": "موافق", "🙏🏻": "رجاء طلب", "😳": "مصدوم", "👏🏼": "تصفيق",
          "🎶": "موسيقي", "🌚": "صمت", "💚": "حب", "🙏": "رجاء طلب", "💘": "حب", "🍃": "سلام", "☺": "يضحك", "🐸": "ضفدع",
          "😶": "مصدوم", "✌": "مرح", "✋🏻": "توقف", "😉": "غمزة", "🌷": "حب", "🙃": "مبتسم", "😫": "حزين", "😨": "مصدوم",
          "🎼 ": "موسيقي", "🍁": "مرح", "🍂": "مرح", "💟": "حب", "😪": "حزن", "😆": "يضحك", "😣": "استياء", "☺": "حب",
          "😱": "كارثة", "😁": "يضحك", "😖": "استياء", "🏃🏼": "يجري", "😡": "غضب", "🚶": "يسير", "🤕": "مرض", "‼": "تعجب",
          "🕊": "طائر", "👌🏻": "احسنت", "❣": "حب", "🙊": "مصدوم", "💃": "سعادة مرح", "💃🏼": "سعادة مرح", "😜": "مرح",
          "👊": "ضربة", "😟": "استياء", "💖": "حب", "😥": "حزن", "🎻": "موسيقي", "✒": "يكتب", "🚶🏻": "يسير", "💎": "الماظ",
          "😷": "وباء مرض", "☝": "واحد", "🚬": "تدخين", "💐": "ورد", "🌞": "شمس", "👆": "الاول", "⚠": "تحذير",
          "🤗": "احتواء", "✖": "غلط", "📍": "مكان", "👸": "ملكه", "👑": "تاج", "✔": "صح", "💌": "قلب", "😲": "مندهش",
          "💦": "ماء", "🚫": "خطا", "👏🏻": "برافو", "🏊": "يسبح", "👍🏻": "تمام", "⭕": "دائره كبيره", "🎷": "ساكسفون",
          "👋": "تلويح باليد", "✌🏼": "علامه النصر", "🌝": "مبتسم", "➿": "عقده مزدوجه", "💪🏼": "قوي", "📩": "تواصل معي",
          "☕": "قهوه", "😧": "قلق و صدمة", "🗨": "رسالة", "❗": "تعجب", "🙆🏻": "اشاره موافقه", "👯": "اخوات", "©": "رمز",
          "👵🏽": "سيده عجوزه", "🐣": "كتكوت", "🙌": "تشجيع", "🙇": "شخص ينحني", "👐🏽": "ايدي مفتوحه", "👌🏽": "بالظبط",
          "⁉": "استنكار", "⚽": "كوره", "🕶": "حب", "🎈": "بالون", "🎀": "ورده", "💵": "فلوس", "😋": "جائع", "😛": "يغيظ",
          "😠": "غاضب", "✍🏻": "يكتب", "🌾": "ارز", "👣": "اثر قدمين", "❌": "رفض", "🍟": "طعام", "👬": "صداقة", "🐰": "ارنب",
          "☂": "مطر", "⚜": "مملكة فرنسا", "🐑": "خروف", "🗣": "صوت مرتفع", "👌🏼": "احسنت", "☘": "مرح", "😮": "صدمة",
          "😦": "قلق", "⭕": "الحق", "✏": "قلم", "ℹ": "معلومات", "🙍🏻": "رفض", "⚪": "نضارة نقاء", "🐤": "حزن", "💫": "مرح",
          "💝": "حب", "🍔": "طعام", "❤": "حب", "✈": "سفر", "🏃🏻‍♀": "يسير", "🍳": "ذكر", "🎤": "مايك غناء", "🎾": "كره",
          "🐔": "دجاجة", "🙋": "سؤال", "📮": "بحر", "💉": "دواء", "🙏🏼": "رجاء طلب", "💂🏿 ": "حارس", "🎬": "سينما",
          "♦": "مرح", "💡": "قكرة", "‼": "تعجب", "👼": "طفل", "🔑": "مفتاح", "♥": "حب", "🕋": "كعبة", "🐓": "دجاجة",
          "💩": "معترض", "👽": "فضائي", "☔": "مطر", "🍷": "عصير", "🌟": "نجمة", "☁": "سحب", "👃": "معترض", "🌺": "مرح",
          "🔪": "سكينة", "♨": "سخونية", "👊🏼": "ضرب", "✏": "قلم", "🚶🏾‍♀": "يسير", "👊": "ضربة", "◾": "وقف", "😚": "حب",
          "🔸": "مرح", "👎🏻": "لا يعجبني", "👊🏽": "ضربة", "😙": "حب", "🎥": "تصوير", "👉": "جذب انتباه", "👏🏽": "يصفق",
          "💪🏻": "عضلات", "🏴": "اسود", "🔥": "حريق", "😬": "عدم الراحة", "👊🏿": "يضرب", "🌿": "ورقه شجره", "✋🏼": "كف ايد",
          "👐": "ايدي مفتوحه", "☠": "وجه مرعب", "🎉": "يهنئ", "🔕": "صامت", "😿": "وجه حزين", "☹": "وجه يائس", "😘": "حب",
          "😰": "خوف و حزن", "🌼": "ورده", "💋": "بوسه", "👇": "لاسفل", "❣": "حب", "🎧": "سماعات", "📝": "يكتب", "😇": "دايخ",
          "😈": "رعب", "🏃": "يجري", "✌🏻": "علامه النصر", "🔫": "يضرب", "❗": "تعجب", "👎": "غير موافق", "🔐": "قفل",
          "👈": "لليمين", "™": "رمز", "🚶🏽": "يتمشي", "😯": "متفاجأ", "✊": "يد مغلقه", "😻": "اعجاب", "🙉": "قرد",
          "👧": "طفله صغيره", "🔴": "دائره حمراء", "💪🏽": "قوه", "💤": "ينام", "👀": "ينظر", "✍🏻": "يكتب", "❄": "تلج",
          "💀": "رعب", "😤": "وجه عابس", "🖋": "قلم", "🎩": "كاب", "☕": "قهوه", "😹": "ضحك", "💓": "حب", "☄ ": "نار",
          "👻": "رعب", "❎": "خطء", "🤮": "حزن", '🏻': "احمر"}
emoticons_to_emoji = {":)": "🙂", ":(": "🙁", "xD": "😆", ":=(": "😭", ":'(": "😢", ":'‑(": "😢", "XD": "😂", ":D": "🙂",
                      "♬": "موسيقي", "♡": "❤", "☻": "🙂"}


def checkemojie(text):
    emojistext = []
    for char in text:
        if any(emoji.distinct_emoji_list(char)) and char in emojis.keys():
            emojistext.append(emojis[emoji.distinct_emoji_list(char)[0]])
    return " ".join(emojistext)


def emojiTextTransform(text):
    cleantext = re.sub(r'[^\w\s]', '', text)
    return cleantext + " " + checkemojie(text)


# Apply checkemojie and emojiTextTransform
df['review_description'] = df['review_description'].apply(lambda x: emojiTextTransform(x))
# print("print data after changing the emoji to text", df['review_description'].head())

# Remove Numbers
df.review_description = df.review_description.apply(lambda x: ''.join([word for word in x if not word.isdigit()]))

# Apply Stemming
arabic_stemmer = ArabicLightStemmer()
# Apply stemming to the 'review_description' column
df['review_description'] = df['review_description'].apply(
    lambda x: " ".join([arabic_stemmer.light_stem(word) for word in x.split()]))


hidden_dim =264 
output_size = 3 
num_layers = 2  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initializations and corrections to your code
df.dropna(subset=['review_description'], inplace=True)
review_description = df['review_description']
y = df['rating']  # 1, 0, -1

tokenizer = Tokenizer()
tokenizer.fit_on_texts(review_description)
seq = tokenizer.texts_to_sequences(review_description)
seq_pad = pad_sequences(seq, maxlen=150, padding="post", truncating="post")
vocab_size = len(tokenizer.word_index) + 1

w2v_embeddings_index = {}
TOTAL_EMBEDDING_DIM = 300
embeddings_file = '/kaggle/working/full_grams_cbow_300_twitter.mdl'
w2v_model = KeyedVectors.load(embeddings_file)

for word in w2v_model.wv.index_to_key:
    w2v_embeddings_index[word] = w2v_model.wv[word]

embedding_matrix = np.zeros((vocab_size, TOTAL_EMBEDDING_DIM))
for word, i in tokenizer.word_index.items():
    embedding_vector = w2v_embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Transformer model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Hyperparameters
max_len = 150
word_index = tokenizer.word_index
vocab_size = len(word_index) + 1
d_model = 150  
num_heads = 5 
num_layers = 2  
dropout_rate = 0.2


# Model definition
class PositionalEncoding(nn.Module):
    def _init_(self, d_model, max_len=150):
        super()._init_()
        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)


        self.register_buffer('pe', pe)

    def forward(self, x):
        x_pe = x + self.pe[:, :x.size(1)]
        return x_pe
    

class transformr_network(nn.Module):
    def _init_(self, vocab_size, d_model, nhead, dim_feedforward, num_layers, dropout, max_len=150):
        super()._init_()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        self.enc_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = TransformerEncoder(self.enc_layers, num_layers)
        self.fc = nn.Linear(d_model, 3)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, src):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = output.mean(dim=1)
        output = self.dropout(output)
        output = self.fc(output)
        return output

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = transformr_network(vocab_size, d_model, num_heads, d_model*4 , num_layers, dropout_rate).to(device)

# Training code
lossfun = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)

from torch.utils.data import random_split

train_size = int(0.8 * len(dataset))
validation_size = len(dataset) - train_size
train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=64, shuffle=False)
for epoch in range(1, 21):
    # Training phase
    model.train() 
    train_loss = 0
    correct_predictions = 0
    total_predictions = 0
    
    for inputs, targets in train_dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = lossfun(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == targets).sum().item()
        total_predictions += targets.size(0)
   
    train_accuracy = 100 * correct_predictions / total_predictions
    print(f"Epoch {epoch}, Training Loss: {train_loss / len(train_dataloader):.4f}, Training Accuracy: {train_accuracy:.2f}%")
    
    # Validation phase
    model.eval() 
    validation_loss = 0
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for inputs, targets in validation_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = lossfun(outputs, targets)
            validation_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == targets).sum().item()
            total_predictions += targets.size(0)
    
    validation_accuracy = 100 * correct_predictions / total_predictions
    print(f"Epoch {epoch}, Validation Loss: {validation_loss / len(validation_dataloader):.4f}, Validation Accuracy: {validation_accuracy:.2f}%")

test=pd.read_csv("/kaggle/input/neural/test _no_label.csv")
test['review_description'] = test['review_description'].apply(remove_duplicate_arabic_words)
test['review_description'] = test['review_description'].apply(
    lambda x: " ".join([word for word in x.split() if word not in stopWords]))
test['review_description'] = test['review_description'].apply(lambda x: emojiTextTransform(x))
test['review_description'] = test['review_description'].apply(lambda x: ''.join([word for word in x if not word.isdigit()]))
test['review_description'] = test['review_description'].apply(
    lambda x: " ".join([arabic_stemmer.light_stem(word) for word in x.split()]))

test_seq = tokenizer.texts_to_sequences(test['review_description'])
test_seq_pad = pad_sequences(test_seq, maxlen=150, padding="post", truncating="post")
X_test = torch.from_numpy(test_seq_pad).long()
X_test = X_test.to(device)

model.eval()
with torch.no_grad():
    output = model(X_test)
    predicted_classes = torch.argmax(output, dim=1)

predicted_labels = le.inverse_transform(predicted_classes.cpu().numpy())
test['predicted_label'] = predicted_labels


test.to_csv('predicted_labels_transform1.csv', columns=['ID', 'review_description', 'predicted_label'], index=False, encoding='utf-8-sig')