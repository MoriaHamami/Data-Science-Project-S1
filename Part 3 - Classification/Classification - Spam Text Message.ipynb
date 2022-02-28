{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "4ea55f47",
   "metadata": {},
   "source": [
    "# Final Project - Part 3 Classification"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f4562f39",
   "metadata": {},
   "source": [
    "##  Project Goal:  classify text messages into either spam or ham "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "badeab50",
   "metadata": {},
   "source": [
    "In this project our goal is to classify text messages into the right class, either spam or ham. \n",
    "\n",
    "We took our data from Kaggle.\n",
    "The websites' link is: https://www.kaggle.com/team-ai/spam-text-message-classification"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "19c4005e",
   "metadata": {},
   "source": [
    "# Table of Contents"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c3b5c5d2",
   "metadata": {},
   "source": [
    "### 1- Checking the Data\n",
    "### 2- Visualization\n",
    "### 3- Cleaning the Data\n",
    "### 4- Splitting the Data\n",
    "### 5-  Dummy Model\n",
    "### 6- Comparing 3 Models\n",
    "### 7- Error Model\n",
    "### 8- Conclusion"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d78179ef",
   "metadata": {},
   "source": [
    "# Checking the Data"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "22f85230",
   "metadata": {},
   "source": [
    "First, let's import all the libraries we need"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "655bff7d",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np \n",
    "import pandas as pd\n",
    "\n",
    "import matplotlib.pyplot as plt\n",
    "import matplotlib as mp\n",
    "import seaborn as sns\n",
    "sns.set(style=\"darkgrid\")\n",
    "\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.dummy import DummyClassifier\n",
    "\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "\n",
    "\n",
    "#for logistic reggression\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.metrics import classification_report, confusion_matrix\n",
    "\n",
    "from sklearn.metrics import roc_auc_score"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "bdeca687",
   "metadata": {},
   "source": [
    "### About our Data.."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d9d8a991",
   "metadata": {},
   "source": [
    "Our data is a DataFrame that has 2 columns:\n",
    "\n",
    "- Category: has the type of the text message either spam or ham\n",
    "- Message: has the content of the text message"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "74932a37",
   "metadata": {},
   "source": [
    "Now, let's import our data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "7d7d36a0",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv ('SPAM text message 20170820 - Data.csv', index_col = False, low_memory=False)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d2379459",
   "metadata": {},
   "source": [
    "A quick look at our data: "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "72e2354c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Category</th>\n",
       "      <th>Message</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>ham</td>\n",
       "      <td>Go until jurong point, crazy.. Available only ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>ham</td>\n",
       "      <td>Ok lar... Joking wif u oni...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>spam</td>\n",
       "      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>ham</td>\n",
       "      <td>U dun say so early hor... U c already then say...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>ham</td>\n",
       "      <td>Nah I don't think he goes to usf, he lives aro...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>spam</td>\n",
       "      <td>FreeMsg Hey there darling it's been 3 week's n...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>ham</td>\n",
       "      <td>Even my brother is not like to speak with me. ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>ham</td>\n",
       "      <td>As per your request 'Melle Melle (Oru Minnamin...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>spam</td>\n",
       "      <td>WINNER!! As a valued network customer you have...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>spam</td>\n",
       "      <td>Had your mobile 11 months or more? U R entitle...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  Category                                            Message\n",
       "0      ham  Go until jurong point, crazy.. Available only ...\n",
       "1      ham                      Ok lar... Joking wif u oni...\n",
       "2     spam  Free entry in 2 a wkly comp to win FA Cup fina...\n",
       "3      ham  U dun say so early hor... U c already then say...\n",
       "4      ham  Nah I don't think he goes to usf, he lives aro...\n",
       "5     spam  FreeMsg Hey there darling it's been 3 week's n...\n",
       "6      ham  Even my brother is not like to speak with me. ...\n",
       "7      ham  As per your request 'Melle Melle (Oru Minnamin...\n",
       "8     spam  WINNER!! As a valued network customer you have...\n",
       "9     spam  Had your mobile 11 months or more? U R entitle..."
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head(10)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b942f64c",
   "metadata": {},
   "source": [
    "The size of our DataFrame"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "b63dc315",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(5572, 2)\n"
     ]
    }
   ],
   "source": [
    "print(df.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "8be5c2c5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead tr th {\n",
       "        text-align: left;\n",
       "    }\n",
       "\n",
       "    .dataframe thead tr:last-of-type th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr>\n",
       "      <th></th>\n",
       "      <th colspan=\"4\" halign=\"left\">Message</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th></th>\n",
       "      <th>count</th>\n",
       "      <th>unique</th>\n",
       "      <th>top</th>\n",
       "      <th>freq</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Category</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>ham</th>\n",
       "      <td>4825</td>\n",
       "      <td>4516</td>\n",
       "      <td>Sorry, I'll call later</td>\n",
       "      <td>30</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>spam</th>\n",
       "      <td>747</td>\n",
       "      <td>641</td>\n",
       "      <td>Please call our customer service representativ...</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "         Message                                                            \\\n",
       "           count unique                                                top   \n",
       "Category                                                                     \n",
       "ham         4825   4516                             Sorry, I'll call later   \n",
       "spam         747    641  Please call our customer service representativ...   \n",
       "\n",
       "               \n",
       "         freq  \n",
       "Category       \n",
       "ham        30  \n",
       "spam        4  "
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.groupby(\"Category\").describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "99315358",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<bound method DataFrame.info of      Category                                            Message\n",
       "0         ham  Go until jurong point, crazy.. Available only ...\n",
       "1         ham                      Ok lar... Joking wif u oni...\n",
       "2        spam  Free entry in 2 a wkly comp to win FA Cup fina...\n",
       "3         ham  U dun say so early hor... U c already then say...\n",
       "4         ham  Nah I don't think he goes to usf, he lives aro...\n",
       "...       ...                                                ...\n",
       "5567     spam  This is the 2nd time we have tried 2 contact u...\n",
       "5568      ham               Will ü b going to esplanade fr home?\n",
       "5569      ham  Pity, * was in mood for that. So...any other s...\n",
       "5570      ham  The guy did some bitching but I acted like i'd...\n",
       "5571      ham                         Rofl. Its true to its name\n",
       "\n",
       "[5572 rows x 2 columns]>"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.info"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "69552b26",
   "metadata": {},
   "source": [
    "Making sure that there are'nt any null values in our data "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "96968ba5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "False"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isnull().values.any()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "962662db",
   "metadata": {},
   "source": [
    "# 2- Visualization\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "99b7d7bf",
   "metadata": {},
   "source": [
    "We added the visualization part in here because later on we will clean our data and by doing so we won't be able to show our raw data as it is."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "ae506378",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 216x360 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.style.use(\"seaborn\")\n",
    "fig, ax = plt.subplots(figsize=(3,5))\n",
    "sns.countplot(x = df.Category)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "1bb0c983",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 576x396 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "label = ['Ham','Spam']\n",
    "colors = sns.color_palette('pastel')[0:5]\n",
    "plt.pie(df['Category'].value_counts(),labels = label, colors = colors, autopct='%.0f%%')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c78c8f6b",
   "metadata": {},
   "source": [
    "Let's check the amount of words in each Category"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "454e04e5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='word_count', ylabel='Category'>"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 576x396 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "df['word_count'] = df['Message'].apply(len)\n",
    "sns.scatterplot(x = df['word_count'],y = df['Category'])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "416101df",
   "metadata": {},
   "source": [
    "As we can see ham type messages are longer than spam type messages. "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "68bf57ad",
   "metadata": {},
   "source": [
    "# 3- Cleaning the Data\n",
    "    \n",
    "   "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "015ee209",
   "metadata": {},
   "source": [
    "We realized that our data is imbalanced because one of our classes has a significantly lower amount of cases compared to the other class (can also be seen in our graphs)."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f1d6b307",
   "metadata": {},
   "source": [
    "In order to balance out the data we'll pick 747 cases of \"ham\" messages."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "424bb5fc",
   "metadata": {},
   "outputs": [],
   "source": [
    "ham=df[df['Category']=='ham']\n",
    "spam=df[df['Category']=='spam']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "a3b28e2b",
   "metadata": {},
   "outputs": [],
   "source": [
    "ham=ham.sample(747)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "89846bc0",
   "metadata": {},
   "source": [
    "Now, let's see the data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "1c192946",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "((747, 3), (747, 3))"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "ham.shape,spam.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "72fad44c",
   "metadata": {},
   "source": [
    "As seen above, our data is now balanced, so it's time to put our samples in our DataFrame "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "f90b0517",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = ham.append(spam,ignore_index=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "a040a427",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Category</th>\n",
       "      <th>Message</th>\n",
       "      <th>word_count</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>ham</td>\n",
       "      <td>It's ok i wun b angry. Msg u aft i come home t...</td>\n",
       "      <td>53</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>ham</td>\n",
       "      <td>tap &amp; spile at seven. * Is that pub on gas st ...</td>\n",
       "      <td>72</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>ham</td>\n",
       "      <td>ZOE IT JUST HIT ME 2 IM FUCKING SHITIN MYSELF ...</td>\n",
       "      <td>103</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>ham</td>\n",
       "      <td>Ok...</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>ham</td>\n",
       "      <td>HELLOGORGEOUS, HOWS U? MY FONE WAS ON CHARGE L...</td>\n",
       "      <td>142</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1489</th>\n",
       "      <td>spam</td>\n",
       "      <td>Want explicit SEX in 30 secs? Ring 02073162414...</td>\n",
       "      <td>90</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1490</th>\n",
       "      <td>spam</td>\n",
       "      <td>ASKED 3MOBILE IF 0870 CHATLINES INCLU IN FREE ...</td>\n",
       "      <td>158</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1491</th>\n",
       "      <td>spam</td>\n",
       "      <td>Had your contract mobile 11 Mnths? Latest Moto...</td>\n",
       "      <td>160</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1492</th>\n",
       "      <td>spam</td>\n",
       "      <td>REMINDER FROM O2: To get 2.50 pounds free call...</td>\n",
       "      <td>147</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1493</th>\n",
       "      <td>spam</td>\n",
       "      <td>This is the 2nd time we have tried 2 contact u...</td>\n",
       "      <td>160</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>1494 rows × 3 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     Category                                            Message  word_count\n",
       "0         ham  It's ok i wun b angry. Msg u aft i come home t...          53\n",
       "1         ham  tap & spile at seven. * Is that pub on gas st ...          72\n",
       "2         ham  ZOE IT JUST HIT ME 2 IM FUCKING SHITIN MYSELF ...         103\n",
       "3         ham                                              Ok...           5\n",
       "4         ham  HELLOGORGEOUS, HOWS U? MY FONE WAS ON CHARGE L...         142\n",
       "...       ...                                                ...         ...\n",
       "1489     spam  Want explicit SEX in 30 secs? Ring 02073162414...          90\n",
       "1490     spam  ASKED 3MOBILE IF 0870 CHATLINES INCLU IN FREE ...         158\n",
       "1491     spam  Had your contract mobile 11 Mnths? Latest Moto...         160\n",
       "1492     spam  REMINDER FROM O2: To get 2.50 pounds free call...         147\n",
       "1493     spam  This is the 2nd time we have tried 2 contact u...         160\n",
       "\n",
       "[1494 rows x 3 columns]"
      ]
     },
     "execution_count": 44,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "356e1603",
   "metadata": {},
   "source": [
    "# 4- Splitting the Data\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "338caa8f",
   "metadata": {},
   "source": [
    "Here we divided our data into a test set (20%) and a train set (80%)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "782483cd",
   "metadata": {},
   "source": [
    "In order to run our codes we changed our strings- ham and spam into numbers:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "7d4a78b4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Category</th>\n",
       "      <th>Message</th>\n",
       "      <th>word_count</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>It's ok i wun b angry. Msg u aft i come home t...</td>\n",
       "      <td>53</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0</td>\n",
       "      <td>tap &amp; spile at seven. * Is that pub on gas st ...</td>\n",
       "      <td>72</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0</td>\n",
       "      <td>ZOE IT JUST HIT ME 2 IM FUCKING SHITIN MYSELF ...</td>\n",
       "      <td>103</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0</td>\n",
       "      <td>Ok...</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0</td>\n",
       "      <td>HELLOGORGEOUS, HOWS U? MY FONE WAS ON CHARGE L...</td>\n",
       "      <td>142</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1489</th>\n",
       "      <td>1</td>\n",
       "      <td>Want explicit SEX in 30 secs? Ring 02073162414...</td>\n",
       "      <td>90</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1490</th>\n",
       "      <td>1</td>\n",
       "      <td>ASKED 3MOBILE IF 0870 CHATLINES INCLU IN FREE ...</td>\n",
       "      <td>158</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1491</th>\n",
       "      <td>1</td>\n",
       "      <td>Had your contract mobile 11 Mnths? Latest Moto...</td>\n",
       "      <td>160</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1492</th>\n",
       "      <td>1</td>\n",
       "      <td>REMINDER FROM O2: To get 2.50 pounds free call...</td>\n",
       "      <td>147</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1493</th>\n",
       "      <td>1</td>\n",
       "      <td>This is the 2nd time we have tried 2 contact u...</td>\n",
       "      <td>160</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>1494 rows × 3 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "      Category                                            Message  word_count\n",
       "0            0  It's ok i wun b angry. Msg u aft i come home t...          53\n",
       "1            0  tap & spile at seven. * Is that pub on gas st ...          72\n",
       "2            0  ZOE IT JUST HIT ME 2 IM FUCKING SHITIN MYSELF ...         103\n",
       "3            0                                              Ok...           5\n",
       "4            0  HELLOGORGEOUS, HOWS U? MY FONE WAS ON CHARGE L...         142\n",
       "...        ...                                                ...         ...\n",
       "1489         1  Want explicit SEX in 30 secs? Ring 02073162414...          90\n",
       "1490         1  ASKED 3MOBILE IF 0870 CHATLINES INCLU IN FREE ...         158\n",
       "1491         1  Had your contract mobile 11 Mnths? Latest Moto...         160\n",
       "1492         1  REMINDER FROM O2: To get 2.50 pounds free call...         147\n",
       "1493         1  This is the 2nd time we have tried 2 contact u...         160\n",
       "\n",
       "[1494 rows x 3 columns]"
      ]
     },
     "execution_count": 45,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Data frame is created under column name Category and Message\n",
    "data_frame = pd.DataFrame(df, columns=[\"Category\", \"Message\"])\n",
    " \n",
    "# Data of Category is converted into Binary Data\n",
    "df_one = pd.get_dummies(data_frame[\"Category\"])\n",
    " \n",
    "# Binary Data is Concatenated into Dataframe\n",
    "df_two = pd.concat((df_one, data_frame), axis=1)\n",
    " \n",
    "# Category column is dropped\n",
    "df_two = df_two.drop([\"Category\"], axis=1)\n",
    " \n",
    "# We want ham =0 and spam =1 So we drop Male column here\n",
    "df_two = df_two.drop([\"ham\"], axis=1)\n",
    " \n",
    "# Rename the Column\n",
    "binary_df = df_two.rename(columns={\"spam\": \"Category\"})\n",
    " \n",
    "#Adding the word count column\n",
    "binary_df['word_count'] = df['Message'].apply(len)\n",
    "\n",
    "# Print the Result\n",
    "binary_df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "04dcdf7a",
   "metadata": {},
   "outputs": [],
   "source": [
    "x_train, x_test, y_train, y_test = train_test_split(binary_df['Message'], binary_df['Category'], test_size = 0.2, random_state=0, shuffle = True, stratify=df['Category'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "4f2e651a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "124                                I said its okay. Sorry\n",
       "662                           It vl bcum more difficult..\n",
       "1007    You have WON a guaranteed £1000 cash or a £200...\n",
       "1151    Do you ever notice that when you're driving, a...\n",
       "1096    This message is brought to you by GMW Ltd. and...\n",
       "                              ...                        \n",
       "1321    Show ur colours! Euro 2004 2-4-1 Offer! Get an...\n",
       "312     BABE !!! I miiiiiiissssssssss you ! I need you...\n",
       "268     That depends. How would you like to be treated...\n",
       "272              Die... Now i have e toot fringe again...\n",
       "934     FREE entry into our £250 weekly comp just send...\n",
       "Name: Message, Length: 1195, dtype: object"
      ]
     },
     "execution_count": 47,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x_train"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "id": "46450374",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "124     0\n",
       "662     0\n",
       "1007    1\n",
       "1151    1\n",
       "1096    1\n",
       "       ..\n",
       "1321    1\n",
       "312     0\n",
       "268     0\n",
       "272     0\n",
       "934     1\n",
       "Name: Category, Length: 1195, dtype: uint8"
      ]
     },
     "execution_count": 48,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_train"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "81b47046",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "510     You only hate me. You can call any but you did...\n",
       "1320    U can WIN £100 of Music Gift Vouchers every we...\n",
       "1168    1st wk FREE! Gr8 tones str8 2 u each wk. Txt N...\n",
       "574     Goodmorning, today i am late for  &lt;DECIMAL&...\n",
       "512     So many people seems to be special at first si...\n",
       "                              ...                        \n",
       "488     My uncles in Atlanta. Wish you guys a great se...\n",
       "1041    FREE2DAY sexy St George's Day pic of Jordan!Tx...\n",
       "1464    u r subscribed 2 TEXTCOMP 250 wkly comp. 1st w...\n",
       "1141    Hey Boys. Want hot XXX pics sent direct 2 ur p...\n",
       "1132    INTERFLORA - It's not too late to order Inter...\n",
       "Name: Message, Length: 299, dtype: object"
      ]
     },
     "execution_count": 49,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x_test"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "id": "be128fa6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "510     0\n",
       "1320    1\n",
       "1168    1\n",
       "574     0\n",
       "512     0\n",
       "       ..\n",
       "488     0\n",
       "1041    1\n",
       "1464    1\n",
       "1141    1\n",
       "1132    1\n",
       "Name: Category, Length: 299, dtype: uint8"
      ]
     },
     "execution_count": 50,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_test"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "09d6a116",
   "metadata": {},
   "source": [
    "# 5- Dummy Model\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2487c49c",
   "metadata": {},
   "source": [
    "A dummy classifier is used as a simple classifier to compare with more complex classifiers."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "id": "d45f3145",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "dummy classifier:0.4983277591973244\n"
     ]
    }
   ],
   "source": [
    "dc_clf = DummyClassifier(strategy=\"most_frequent\")\n",
    "\n",
    "dc_clf.fit(x_train,y_train)\n",
    "score = dc_clf.score(x_test,y_test)\n",
    "print(\"dummy classifier:{}\".format(score))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "15ea7adf",
   "metadata": {},
   "source": [
    "Our dummy classifier gives us accuracy of 49%, thats around 50%. \n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b69ce84f",
   "metadata": {},
   "source": [
    "# 6- Comparing 3 Models\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "84e58b17",
   "metadata": {},
   "source": [
    "### Our data in this project is type string so we used models that work on type string data "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "50d11a34",
   "metadata": {},
   "source": [
    "#### First model: Random Forest Classifier:"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "726ffd1b",
   "metadata": {},
   "source": [
    "The randomforest classifier has multiple decision trees. In order to enhance its accuracy and avoid overfitting it uses randomness.\n",
    "the algorithm makes decision trees based on a random selection of data samples and get predictions from every tree. After that, they select the best viable solution through votes. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "dc7a00cc",
   "metadata": {},
   "outputs": [],
   "source": [
    "clf = Pipeline([('tfidf', TfidfVectorizer()), ('clf', RandomForestClassifier(n_estimators=100, n_jobs=-1))]) \n",
    "#njobs=-1 will use all the cores of CPU"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "d7a90656",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Pipeline(steps=[('tfidf', TfidfVectorizer()),\n",
       "                ('clf', RandomForestClassifier(n_jobs=-1))])"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "clf.fit(x_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "c04a07e6",
   "metadata": {},
   "outputs": [],
   "source": [
    "rfc_pred = clf.predict(x_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "ec0ad91c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[150,   0],\n",
       "       [ 16, 133]], dtype=int64)"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.metrics import accuracy_score, classification_report, confusion_matrix\n",
    "confusion_matrix(y_test, rfc_pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "6388bc4b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.90      1.00      0.95       150\n",
      "           1       1.00      0.89      0.94       149\n",
      "\n",
      "    accuracy                           0.95       299\n",
      "   macro avg       0.95      0.95      0.95       299\n",
      "weighted avg       0.95      0.95      0.95       299\n",
      "\n"
     ]
    }
   ],
   "source": [
    "print(classification_report(y_test, rfc_pred))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3527216b",
   "metadata": {},
   "source": [
    "#### Second model: Support Vector Machines:"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "289af556",
   "metadata": {},
   "source": [
    "The support vector machine algorithm helps to classify data points with linear classification. \n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "669e56ba",
   "metadata": {},
   "source": [
    "In the SVM algorithm, we plot each data item as a point in n-dimensional space (where n is a number of features you have) with the value of each feature being the value of a particular coordinate. Then, we perform classification by finding the hyper-plane (a subspace whose dimension is one less than that of its surrounding space) that differentiates the two classes very well"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "53dcbe60",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.svm import SVC"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "aff4d020",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Pipeline(steps=[('tfidf', TfidfVectorizer()),\n",
       "                ('clf', SVC(C=1000, gamma='auto'))])"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "clf = Pipeline([('tfidf', TfidfVectorizer()), ('clf', SVC(C = 1000, gamma = 'auto'))])\n",
    "clf.fit(x_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "id": "369a8a0d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[150,   0],\n",
       "       [ 14, 135]], dtype=int64)"
      ]
     },
     "execution_count": 55,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "svc_pred = clf.predict(x_test)\n",
    "confusion_matrix(y_test, svc_pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "id": "86787510",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.91      1.00      0.96       150\n",
      "           1       1.00      0.91      0.95       149\n",
      "\n",
      "    accuracy                           0.95       299\n",
      "   macro avg       0.96      0.95      0.95       299\n",
      "weighted avg       0.96      0.95      0.95       299\n",
      "\n"
     ]
    }
   ],
   "source": [
    "print(classification_report(y_test, svc_pred))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9243e656",
   "metadata": {},
   "source": [
    "#### Third model: Logistic Regression"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "504eacaf",
   "metadata": {},
   "source": [
    "Logistic regression is a statistical analysis method to predict a binary outcome, such as spam or ham, based on prior observations of a data set. A logistic regression model predicts a dependent data variable by analyzing the relationship between one or more existing independent variables."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "id": "9a4d6014",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Pipeline(steps=[('tfidf', TfidfVectorizer()), ('clf', LogisticRegression())])"
      ]
     },
     "execution_count": 58,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "clf = Pipeline([('tfidf', TfidfVectorizer()), ('clf', LogisticRegression())])\n",
    "clf.fit(x_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "id": "936239f2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[150,   0],\n",
       "       [ 14, 135]], dtype=int64)"
      ]
     },
     "execution_count": 59,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "l_pred = clf.predict(x_test)\n",
    "confusion_matrix(y_test, l_pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "id": "d105a104",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.91      1.00      0.96       150\n",
      "           1       1.00      0.91      0.95       149\n",
      "\n",
      "    accuracy                           0.95       299\n",
      "   macro avg       0.96      0.95      0.95       299\n",
      "weighted avg       0.96      0.95      0.95       299\n",
      "\n"
     ]
    }
   ],
   "source": [
    "print(classification_report(y_test, l_pred))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ce9fab44",
   "metadata": {},
   "source": [
    "As we can see our first model is not performing as good as the other two, therefore our best models are logistic regression and support vector machines."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a6cfdd29",
   "metadata": {},
   "source": [
    "The reason that our models are so close in accuracy is because we don't have enough data for the models to work on, and that's the reason they all work the same and get the same answers.\n",
    "\n",
    "However, we can see that in our second model: Support Vector Machines, the clasisification report came out different, but the accuracy is similar to the other two models. "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cc75190a",
   "metadata": {},
   "source": [
    "# 7- Error Model\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2c8e0e4c",
   "metadata": {},
   "source": [
    "We decided to use AUC (Area Under The Curve) ROC (Receiver Operating Characteristics) in order to score our models.\n",
    "\n",
    "When AUC = 1, then the classifier is able to perfectly distinguish between all the Positive and the Negative class points correctly.\n",
    "\n",
    "When 0.5<AUC<1, there is a high chance that the classifier will be able to distinguish the positive class values from the negative class values.\n",
    "This is so because the classifier is able to detect more numbers of True positives and True negatives than False negatives and False positives.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ae8370de",
   "metadata": {},
   "source": [
    "#### Random Forest Classifier:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "id": "c5e3be0b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9463087248322148"
      ]
     },
     "execution_count": 62,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "roc_auc_score(y_test,rfc_pred)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9d6f7b60",
   "metadata": {},
   "source": [
    "####  Support Vector Machines:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "id": "db6d8ab2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9530201342281879"
      ]
     },
     "execution_count": 63,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "roc_auc_score(y_test,svc_pred)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "405d6cb5",
   "metadata": {},
   "source": [
    "#### Logistic Regression:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "id": "c2549966",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9530201342281879"
      ]
     },
     "execution_count": 64,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "roc_auc_score(y_test,l_pred)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4f1f257e",
   "metadata": {},
   "source": [
    "# 8- Conclusion"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "79d5082e",
   "metadata": {},
   "source": [
    "When AUC=0.5, then the classifier is not able to distinguish between Positive and Negative class points. Meaning either the classifier is predicting random class or constant class for all the data points. Therefore, we can see that our Dummy Model worked accordingly. "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a43e82f0",
   "metadata": {},
   "source": [
    "In comparison, we saw that the Logistic Regression and Support Vector Machines were the best models, both giving us 94.6% ~95% accuracy. "
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
