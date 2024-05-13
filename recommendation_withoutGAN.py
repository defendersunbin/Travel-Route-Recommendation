#withoutGAN

import pandas as pd
import numpy as np
import stellargraph as sg
import matplotlib.pyplot as plt
import os
import pickle
import random

# 결과 데이터를 불러오는 함수
def load_result_data():
    with open('result_data.pkl', 'rb') as f:
        return pickle.load(f)

# 'result_data.pkl' 파일이 존재하면 데이터 불러오기
if os.path.exists('result_data.pkl'):
    combined_df_with_scores = load_result_data()
    print("저장된 결과 데이터를 불러왔습니다.")
else:
    # 파일이 없으면 기존 코드 실행하여 데이터 생성
    print("저장된 결과 데이터가 없습니다. 코드를 실행하여 데이터를 생성합니다.")

route_review = pd.read_csv('tripadvisor_route_review.csv')
route = pd.read_csv('tripadvisor_route_list.csv')
temp = route.merge(route_review, left_on='ROUTENAME', right_on='ROUTE_ID')

# Calculate average rating for each route
average_ratings = route_review.groupby('ROUTE_ID')['Rating'].mean()

# 각 ROUTE_ID에 대한 총 합산 점수 계산
total_scores = route_review.groupby('ROUTE_ID')['Rating'].sum()

# Define thresholds for recommendation
# These thresholds are arbitrary for demonstration; adjust them as needed.
most_recommended_threshold = 4 # Top 25%
not_recommended_threshold = 3.9  # Bottom 25%


#node
route_node=temp['ROUTE_ID']
place_node=temp['Place']
user_node=temp['USER_ID']

route_node = route_node.drop_duplicates()
place_node = place_node.drop_duplicates()
user_node = user_node.drop_duplicates()

route_node_ids=pd.DataFrame(route_node)
place_node_ids=pd.DataFrame(place_node)
user_node_ids=pd.DataFrame(user_node)

route_node_ids.set_index('ROUTE_ID', inplace=True)
place_node_ids.set_index('Place', inplace=True)
user_node_ids.set_index('USER_ID', inplace=True)

#edge
user_route_edge = temp[['USER_ID', 'ROUTE_ID']]
user_route_edge.columns = ['source', 'target']

route_place_edge = temp[['ROUTE_ID', 'Place']]
route_place_edge.columns = ['source','target']

start=len(user_route_edge)
route_place_edge.index=range(start, start+len(route_place_edge))

g=sg.StellarDiGraph(nodes={'user' : user_node_ids, 'route' : route_node_ids, 'place' : place_node_ids},
                    edges={'user_route' : user_route_edge, 'route_place' : route_place_edge})

print(g.info())


#HIN 임베딩
walk_length = 50
metapaths = [["user", "route", "place", "route", "user"], ["user", "route", "user"]]


from stellargraph.data import UniformRandomMetaPathWalk

rw = UniformRandomMetaPathWalk(g)

walks = rw.run(
    nodes=list(g.nodes()),  # root nodes
    length=walk_length,  # maximum length of a random walk
    n=10,  # number of random walks per root node
    metapaths=metapaths,  # the metapaths
    seed=42
)

#Route Embedding

course_sequence = pd.read_csv('tripadvisor_route_list.csv', encoding = 'UTF-8')

course_sequence.columns=["ROUTENAME", "Place"]
course_sequence_nan = course_sequence[course_sequence['Place'].str.contains("nan", na = True, case=False)]
course_sequence = course_sequence[course_sequence['Place'].isin(course_sequence_nan['Place'])== False]

places = (course_sequence['Place'])

# 단어 목록을 인덱스로 매핑하는 딕셔너리 생성
word_to_index = {}
index_to_word = {}
current_index = 0

# 장소 데이터를 단어 인덱스의 시퀀스로 변환
sequences = []
for place in places:
    sequence = []
    for word in place.split(", "):
        if word not in word_to_index:
            word_to_index[word] = current_index
            index_to_word[current_index] = word
            current_index += 1
        sequence.append(word_to_index[word])
    sequences.append(sequence)

from sklearn.model_selection import train_test_split

# 필요한 특성 선택
features = route_review[['USER_ID', 'ROUTE_ID', 'Rating']]

# 'Rating'을 레이블로 사용
X = features.drop('Rating', axis=1)
y = features['Rating']

# 데이터를 훈련 세트와 테스트 세트로 분할
training_data, test_data, training_labels, test_labels = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Initialize LabelEncoder
le_user = LabelEncoder()
le_route = LabelEncoder()

# Fit LabelEncoder on the entire dataset before splitting (to avoid data leakage, this is just illustrative)
all_user_ids = pd.concat([route_review['USER_ID'] for route_review in [training_data, test_data]])
all_route_ids = pd.concat([route_review['ROUTE_ID'] for route_review in [training_data, test_data]])

le_user.fit(all_user_ids)
le_route.fit(all_route_ids)

# Transform both training and test data
training_data['USER_ID'] = le_user.transform(training_data['USER_ID'])
test_data['USER_ID'] = le_user.transform(test_data['USER_ID'])

training_data['ROUTE_ID'] = le_route.transform(training_data['ROUTE_ID'])
test_data['ROUTE_ID'] = le_route.transform(test_data['ROUTE_ID'])

# Now apply StandardScaler
scaler = StandardScaler()
training_data_normalized = scaler.fit_transform(training_data)
test_data_normalized = scaler.transform(test_data)

# 데이터 정규화
scaler = StandardScaler()
training_data_normalized = scaler.fit_transform(training_data)
test_data_normalized = scaler.transform(test_data)

# MLP Regressor 초기화 및 훈련
mlp = MLPRegressor(hidden_layer_sizes=(200,150,100,50), max_iter=100, alpha=1e-5, solver='adam', verbose=0, random_state=150, learning_rate_init=0.001)
mlp.fit(training_data_normalized, training_labels)

# 테스트 데이터에 대한 예측
mlp_pred = mlp.predict(test_data_normalized)

# RMSE와 MAE 계산
rmse = mean_squared_error(mlp_pred, test_labels)**0.5
mae = mean_absolute_error(mlp_pred, test_labels)

print(rmse)
print(mae)

#Data for plotting
errors = ['rmse', 'mae']
values = [rmse, mae]

# 결과 시각화
plt.figure(figsize=(8,5))
plt.bar(['RMSE', 'MAE'], [rmse, mae], color=['blue', 'orange'])
plt.xlabel('Error Type')
plt.ylabel('Value')
plt.title('RMSE and MAE Visualization')
plt.ylim(0, max([rmse, mae]) + 0.1 * max([rmse, mae]))
plt.show()

# Identify Most and Not Recommended Routes
most_recommended_routes = average_ratings[average_ratings >= most_recommended_threshold].index.tolist()
not_recommended_routes = average_ratings[average_ratings <= not_recommended_threshold].index.tolist()

print("Most Recommended Routes: \n", most_recommended_routes)
print("Not Recommended Routes: \n", not_recommended_routes)

# 'most_recommended_routes'와 'not_recommended_routes' 리스트를 데이터프레임으로 변환
most_recommended_df = pd.DataFrame(most_recommended_routes, columns=['ROUTE_ID'])
most_recommended_df['Recommendation'] = 'Most Recommended'

not_recommended_df = pd.DataFrame(not_recommended_routes, columns=['ROUTE_ID'])
not_recommended_df['Recommendation'] = 'Not Recommended'

# Most Recommended와 Not Recommended 데이터프레임에 합산 점수 추가
most_recommended_df['Total_Score'] = most_recommended_df['ROUTE_ID'].map(total_scores)
not_recommended_df['Total_Score'] = not_recommended_df['ROUTE_ID'].map(total_scores)

# Most Recommended와 Not Recommended 데이터프레임에 평균 평점 추가
most_recommended_df['Average_Rating'] = most_recommended_df['ROUTE_ID'].map(average_ratings)
not_recommended_df['Average_Rating'] = not_recommended_df['ROUTE_ID'].map(average_ratings)

# 두 데이터프레임을 하나로 합치기
combined_df_with_scores = pd.concat([most_recommended_df, not_recommended_df], ignore_index=True)

# CSV 파일로 저장
combined_df_with_scores.to_csv('recommended_routes_with_scores_withoutGAN.csv', index=False)

print("CSV 파일이 성공적으로 저장되었습니다.")

# 단계 1: 'Most Recommended' 경로 추출
combined_df_with_scores = pd.read_csv('recommended_routes_with_scores_withoutGAN.csv')
most_recommended_routes = combined_df_with_scores[combined_df_with_scores['Recommendation'] == 'Most Recommended']

# 단계 2: 경로에 해당하는 'Place' 찾기
route_list = pd.read_csv('tripadvisor_route_list.csv')
recommended_places = route_list[route_list['ROUTENAME'].isin(most_recommended_routes['ROUTE_ID'])]

# 단계 3: 무작위로 'Place' 추천
places_to_recommend = recommended_places['Place'].unique()
number_of_places_to_recommend = random.randint(4, 8) # 4에서 8개 사이의 숫자를 무작위로 선택
recommended_places_list = random.sample(list(places_to_recommend), number_of_places_to_recommend)

# 추천된 장소를 데이터프레임으로 변환하고 CSV 파일로 저장
recommended_places_df = pd.DataFrame(recommended_places_list, columns=['Recommended Places'])
recommended_places_df.to_csv('recommended_places_withoutGAN.csv', index=False)

print("추천 장소: ", recommended_places_list)
print("CSV 파일이 'recommended_places_withoutGAN.csv'로 저장되었습니다.")

# recommended_places.csv 파일 읽기
places_df = pd.read_csv('recommended_places_withoutGAN.csv')

import pickle

with open('result_data.pkl', 'wb') as f:
    pickle.dump(combined_df_with_scores, f)

print("결과 데이터가 'result_data.pkl' 파일로 저장되었습니다.")