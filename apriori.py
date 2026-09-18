import time
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from collections import defaultdict

def load_transactions(filepath):
    transactions = []
    with open(filepath, 'r', encoding='cp1251') as f:
        for line in f:
            items = [item.strip() for item in line.strip().split(',') if item.strip()]
            if items:
                transactions.append(set(items))
    return transactions

def get_frequent_1_itemsets(transactions, min_support):
    """Поиск частых 1-элементных наборов."""
    n_trans = len(transactions)
    item_counts = defaultdict(int)
    for trans in transactions:
        for item in trans:
            item_counts[item] += 1
            
    frequent_items = {}
    for item, count in item_counts.items():
        support = count / n_trans
        if support >= min_support:
            frequent_items[frozenset([item])] = support
    return frequent_items


def generate_candidates(frequent_k_minus_1, k):
    candidates = set()
    itemsets = list(frequent_k_minus_1.keys())
    n = len(itemsets)
    
    for i in range(n):
        for j in range(i + 1, n):
            union_set = itemsets[i] | itemsets[j]
            if len(union_set) == k:
                # Проверка: все подмножества длины k-1 должны быть частыми
                all_subsets_frequent = True
                for sub in combinations(union_set, k - 1):
                    if frozenset(sub) not in frequent_k_minus_1:
                        all_subsets_frequent = False
                        break
                if all_subsets_frequent:
                    candidates.add(union_set)
    return candidates


def apriori(transactions, min_support=0.05, sort_by='support_desc'):
    n_trans = len(transactions)
    all_frequent = {}
    
    current_frequent = get_frequent_1_itemsets(transactions, min_support)
    all_frequent.update(current_frequent)
    
    k = 2
    while current_frequent:
        candidates = generate_candidates(current_frequent, k)
        if not candidates:
            break
            
        counts = defaultdict(int)
        for trans in transactions:
            for cand in candidates:
                if cand.issubset(trans):
                    counts[cand] += 1
                    
        current_frequent = {}
        for cand, count in counts.items()ы:
            sup = count / n_trans
            if sup >= min_support:
                current_frequent[cand] = sup
                
        all_frequent.update(current_frequent)
        k += 1
        
    result = list(all_frequent.items())

    if sort_by == 'support_desc':
        result.sort(key=lambda x: (-x[1], sorted(list(x[0]))))
    elif sort_by == 'lexicographical':
        result.sort(key=lambda x: sorted(list(x[0])))
    else:
        raise ValueError("sort_by должен быть 'support_desc' или 'lexicographical'")
        
    return result

def run_experiments(filepath):
    transactions = load_transactions(filepath)
    thresholds = [0.01, 0.03, 0.05, 0.10, 0.15]
    
    execution_times = []
    itemsets_by_length = {k: [] for k in range(1, 5)}  # длины от 1 до 4+
    total_counts = []
    
    print(f"Всего загружено транзакций: {len(transactions)}\n")
    print("Результаты запусков:")
    print("Порог (min_sup) | Время (с) | Всего наборов | Распределение по длинам (1, 2, 3, 4+)")
    print("-" * 75)
    
    for sup in thresholds:
        start_t = time.perf_counter()
        frequent = apriori(transactions, min_support=sup, sort_by='support_desc')
        elapsed_t = time.perf_counter() - start_t
        execution_times.append(elapsed_t)
        total_counts.append(len(frequent))
        
        counts_len = defaultdict(int)
        for itemset, _ in frequent:
            counts_len[len(itemset)] += 1
            
        for k in itemsets_by_length:
            itemsets_by_length[k].append(counts_len[k])
            
        lengths_str = f"L1: {counts_len[1]}, L2: {counts_len[2]}, L3: {counts_len[3]}, L4: {counts_len[4]}"
        print(f"{sup * 100:5.1f}%          | {elapsed_t:9.4f} | {len(frequent):13d} | {lengths_str}")

    threshold_labels = [f"{int(t*100)}%" for t in thresholds]
    
    plt.figure(figsize=(8, 5))
    plt.plot(threshold_labels, execution_times, marker='o', linewidth=2, color='#1f77b4')
    plt.title('Зависимость времени выполнения Apriori от порога поддержки', fontsize=12)
    plt.xlabel('Минимальный порог поддержки (min_support)', fontsize=11)
    plt.ylabel('Время выполнения, с', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    for i, txt in enumerate(execution_times):
        plt.annotate(f"{txt:.2f} с", (threshold_labels[i], execution_times[i]),
                     textcoords="offset points", xytext=(0, 7), ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig('performance_vs_support.png', dpi=300)
    plt.close()
    
    df_counts = pd.DataFrame(itemsets_by_length, index=threshold_labels)

    df_counts = df_counts.loc[:, (df_counts != 0).any(axis=0)]
    df_counts.rename(columns=lambda c: f'Длина {c}', inplace=True)

    fig, ax = plt.subplots(figsize=(11, 6))

    colors = ['#2b5c8f', '#e67e22', '#27ae60', '#8e44ad']
    bars = df_counts.plot(
        kind='bar', 
        stacked=False, 
        ax=ax, 
        width=0.75, 
        edgecolor='black',
        linewidth=0.8,
        color=colors[:len(df_counts.columns)]
    )

    plt.title('Количество частых наборов по длинам при различных порогах поддержки', fontsize=13, pad=15)
    plt.xlabel('Минимальный порог поддержки (min_support)', fontsize=11, labelpad=10)
    plt.ylabel('Количество частых наборов (шт.)', fontsize=11)
    plt.xticks(rotation=0, fontsize=10)
    plt.legend(title='Размер набора (k-itemset)', fontsize=10, title_fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    for container in ax.containers:
        labels = [f'{int(v):d}' if v > 0 else '' for v in container.datavalues]
        ax.bar_label(container, labels=labels, padding=3, fontsize=9, fontweight='bold')

    ax.set_ylim(0, df_counts.values.max() * 1.15)

    plt.tight_layout()
    plt.savefig('itemsets_length_vs_support_grouped.png', dpi=300)
    
    print("\nГрафики успешно сохранены: 'performance_vs_support.png' и 'itemsets_length_vs_support.png'.")

if __name__ == '__main__':
    run_experiments('./BigSmoke(Data)/baskets.csv')