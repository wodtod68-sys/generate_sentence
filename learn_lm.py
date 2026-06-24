#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import pickle # dump
from collections import defaultdict

# 2gram 언어 모델 학습
# unigram 빈도와 bigram 빈도를 리턴
# 문장의 앞뒤에 시작(<s>)과 끝(</s>)을 나타내는 가상의 단어를 포함해야 함
# 단어 토큰은 공백을 기준으로 분리
def learn_bigram_language_model(input_file):
    # 단어 빈도를 저장할 dictionary
    unigram_counts = defaultdict(int)
    bigram_counts = {} # dictionary of dictionary
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # 문장 앞뒤에 <s>, </s> 추가 및 띄어쓰기 기준 토큰화
            tokens = ['<s>'] + line.split() + ['</s>']
            
            # 빈도 누적
            for i in range(len(tokens)):
                unigram_counts[tokens[i]] += 1
                
                if i < len(tokens) - 1:
                    w1, w2 = tokens[i], tokens[i+1]
                    if w1 not in bigram_counts:
                        bigram_counts[w1] = defaultdict(int)
                    bigram_counts[w1][w2] += 1
                    
    # 일반 dictionary로 형변환 (선택 사항이지만 안전한 데이터 직렬화를 위해 적용)
    unigram_counts = dict(unigram_counts)
    for w1 in bigram_counts:
        bigram_counts[w1] = dict(bigram_counts[w1])
        
    return unigram_counts, bigram_counts
    

################################################################################
def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} input_file output_file(pickle)")
        sys.exit(1)
        
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    # 모델 학습
    unigram_counts, bigram_counts = learn_bigram_language_model(input_file)
    
    # 모델 저장
    model = {
        'unigram_counts': unigram_counts,
        'bigram_counts': bigram_counts
    }
    
    with open(output_file, 'wb') as f:
        pickle.dump(model, f)

################################################################################
if __name__ == "__main__":
    main() 