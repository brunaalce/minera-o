# Passo 1: Importar bibliotecas necessárias
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Passo 2: Carregar o dataset
# Aqui, carregamos o dataset a partir de um arquivo CSV. O arquivo pode ter transações de itens de um mercado ou qualquer outro tipo de transação.
# Caso o arquivo esteja em outro formato, você pode adaptá-lo conforme necessário.

# Substitua 'path_to_your_dataset.csv' pelo caminho do seu arquivo
dataset = pd.read_csv('path_to_your_dataset.csv', header=None)

# Visualizar as primeiras linhas do dataset para verificar como os dados estão organizados
print(dataset.head())

# Passo 3: Pré-processamento dos dados
# O Apriori espera os dados em um formato binário, onde cada transação é representada por uma lista de itens (1 significa que o item foi comprado, 0 que não foi).
# Aqui, usamos o TransactionEncoder para transformar os dados para o formato correto.

# Convertendo o dataset para uma lista de transações (listas de itens)
transactions = dataset.values.tolist()

# Inicializamos o TransactionEncoder e transformamos as transações em um formato binário (matriz de 1s e 0s)
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)

# Criamos um DataFrame com os dados binários
df = pd.DataFrame(te_ary, columns=te.columns_)

# Visualizar os dados no formato binário
print(df.head())

# Passo 4: Aplicar o algoritmo Apriori
# O algoritmo Apriori é usado para encontrar os itemsets frequentes, ou seja, conjuntos de itens que aparecem frequentemente juntos nas transações.
# O parâmetro min_support determina o suporte mínimo para que um itemset seja considerado frequente. Ajuste esse valor conforme necessário.

# Aqui, definimos min_support=0.05, ou seja, um itemset deve aparecer em pelo menos 5% das transações.
frequent_itemsets = apriori(df, min_support=0.05, use_colnames=True)

# Visualizar os itemsets frequentes
print(frequent_itemsets)

# Passo 5: Gerar regras de associação
# Após encontrar os itemsets frequentes, podemos gerar as regras de associação. As regras indicam como a presença de um conjunto de itens pode implicar na presença de outro conjunto de itens.

# Usamos a métrica "lift", que mede a relevância das regras. O lift maior que 1 indica uma relação interessante entre os itens.
# O min_threshold=1 significa que queremos regras com lift maior ou igual a 1.

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

# Visualizar as regras geradas
print(rules)

# Passo 6: Filtrar regras com alta confiança
# Podemos filtrar as regras que possuem alta confiança. A confiança é uma medida de quão frequentemente os itens do consequente aparecem nas transações que contêm o antecedente.
# Aqui, estamos filtrando regras com confiança maior que 0.7, o que significa que, se o antecedente ocorrer, o consequente ocorrerá em 70% das vezes ou mais.

high_confidence_rules = rules[rules['confidence'] > 0.7]
print(high_confidence_rules)

# Passo 7: Explorar diferentes parâmetros e possibilidades
# Agora que temos a base, podemos explorar diferentes parâmetros e possibilidades.

# 1. Alterar o valor de min_support:
# Testar diferentes valores de min_support pode afetar o número de itemsets frequentes encontrados. Por exemplo, use 0.1 (10%) para ver um subconjunto maior de itemsets.
frequent_itemsets = apriori(df, min_support=0.1, use_colnames=True)
print(frequent_itemsets)

# 2. Usar diferentes métricas para as regras:
# Ao invés de "lift", podemos usar outras métricas como "confidence", que é mais direta para medir a relação entre os itens.
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
print(rules)

# 3. Ajustar o min_threshold para as regras de associação:
# O min_threshold ajusta o critério mínimo para gerar as regras. Se aumentar o valor, você verá menos regras, mas com maior qualidade.
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.5)
print(rules)

# Passo 8: Visualização das Regras
# Você pode visualizar as regras de uma forma mais interativa, como gráficos de rede ou outros métodos, para melhor entender as relações.
# Mas isso pode exigir bibliotecas adicionais como matplotlib ou networkx.
