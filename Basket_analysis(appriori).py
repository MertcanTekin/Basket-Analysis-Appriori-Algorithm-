import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx


# CSV dosyasını yükleme
data = pd.read_csv("C:/Users/user/Desktop/Assignment-1_Data2.csv")

# date sütununu date formatına çevirdik
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

# "A" ile başlayan BillNo değerlerini veri setinden çıkarma
data_cleaned = data[~data['BillNo'].astype(str).str.startswith('A')]

# "Quantity" sütunu 0 ve 0'dan küçük olan satırları çıkaralım
data_cleaned = data_cleaned[data_cleaned['Quantity'] > 0]

#eksik verielri siler
data_cleaned = data_cleaned.dropna()


# BillNo'ya göre gruplama ve her bir fatura için satın alınan ürünleri listeleme
basket = data_cleaned.groupby('BillNo')['Itemname'].apply(list)

# basket=list(basket["Itemname"].apply(lambda x:x.split(",")))

# Basket analizi için uygun formata getirme
basket_sets = basket.str.join(',').str.get_dummies(sep=',')

# # Sıkça satın alınan ürünleri bulma (minimum destek değeri 0.02 olarak belirlendi)
frequent_itemsets = apriori(basket_sets, min_support=0.02, use_colnames=True)




# # Ürünler arasındaki ilişkileri bulma
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)




# # Sonuçları confidence değerine göre sıralama
rules_sorted = rules.sort_values(by="confidence", ascending=False)

#değişkeni csv dosyasına aktarır
# rules_sorted.to_csv("C:/Users/user/Desktop/rulessorted.csv", index=False)



# Heatmap için pivot tablo oluşturma
pivot = rules_sorted.pivot(index='antecedents', columns='consequents', values='confidence')

# Heatmap oluşturma
plt.figure(figsize=(12, 10))
sns.heatmap(pivot, annot=True, cmap='YlGnBu',linewidths=0.5, linecolor='black')
plt.title("Confidence Değerlerine Göre Heatmap")
plt.show()


# Her bir ürün için support değerlerini bulma
single_item_support = frequent_itemsets[frequent_itemsets['itemsets'].apply(lambda x: len(x) == 1)]

# En yüksek support değerine sahip ilk 10 ürünü seçme
top_single_item_support = single_item_support.nlargest(10, 'support')

# Bar grafiğini oluşturma
plt.figure(figsize=(12, 8))
sns.barplot(data=top_single_item_support, y=top_single_item_support['itemsets'].astype(str), x='support', palette='viridis')
plt.title('En Yüksek Support Değerine Sahip İlk 10 Ürün')
plt.xlabel('Support')
plt.ylabel('Ürünler')
plt.xticks(rotation=45)
plt.show()



# Network graph oluşturma
G_adjusted = nx.Graph()

# "antecedents" ve "consequents" sütunlarındaki ürünleri ve ilişkileri ekleyerek düğümler ve kenarlar oluşturma
# Ancak confidence değeri 0.50'den yüksek olanlar için
for _, row in rules_sorted[rules_sorted['confidence'] > 0.5].iterrows():
    G_adjusted.add_edge(', '.join(list(row['antecedents'])), ', '.join(list(row['consequents'])))

# Grafiği görselleştirme için ayarlamalar
plt.figure(figsize=(18, 18))
pos_adjusted = nx.spring_layout(G_adjusted)
nx.draw_networkx(G_adjusted, pos_adjusted, with_labels=True, 
                 node_size=2500,  # Düğüm boyutu
                 node_color='skyblue', 
                 font_size=12,  # Etiket boyutu
                 alpha=0.6, 
                 edge_color='gray')

plt.title("Confidence Değeri 0.50'den Yüksek Ürün Kombinasyonları - Network Graph")
plt.show()
