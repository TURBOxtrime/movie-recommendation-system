import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
df = pd.read_csv('[provide the path for movies.csv file ! ]')
features = ['keywords', 'cast', 'genres', 'director']
for feature in features:
    df[feature] = df[feature].fillna('')
    
def combine_features(row):
    return row['keywords'] + " " + row['cast'] + " " + row['genres'] + " " + row['director']

df['combined_features'] = df.apply(combine_features, axis=1)
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df['combined_features'])
cosine_similarities = cosine_similarity(tfidf_matrix)
def recommend(movie_title):
    movie_index = df[df['title'] == movie_title].index[0]
    similar_movies = list(enumerate(cosine_similarities[movie_index]))
    sorted_similar_movies = sorted(similar_movies, key=lambda x:x[1], reverse=True)
    recommended_movies = []
    for i in sorted_similar_movies[1:6]:
        recommended_movies.append(df.iloc[i[0]]['title'])
    return recommended_movies
print(recommend('[write the name of the movie for similar recommendations]'))
