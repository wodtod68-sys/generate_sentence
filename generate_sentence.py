#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle # load
import random # choice, choices
import sys
import math # log

################################################################################
# 빈도를 가중치로 적용하여 다음 단어 선택 (random.choices 사용)
# 현재 단어가 모델에 없는 경우, 유니그램에서 랜덤하게 단어 선택
def get_next_word(model, current_word):
    bigram = model['bigram_counts']
    unigram = model['unigram_counts']
    
    # 1. 현재 단어가 2-gram 모델에 존재하는 경우 (가중치 적용)
    if current_word in bigram and bigram[current_word]:
        next_words = list(bigram[current_word].keys())
        weights = list(bigram[current_word].values())
        return random.choices(next_words, weights=weights, k=1)[0]
    
    # 2. 현재 단어가 모델에 없는 경우 유니그램에서 랜덤하게 단어 선택
    else:
        if unigram:
            # random.choice를 사용하여 유니그램(전체 단어) 중 하나를 랜덤 추출
            return random.choice(list(unigram.keys()))
        else:
            return '</s>'

################################################################################
# 문장의 로그 확률 계산 (로그 취한 개별 확률들의 합)
# 모델에 없는 단어 또는 단어 바이그램이 있으면 -100을 더함
def get_probability(model, sentence):
    # 로그 확률 초기화
    log_prob = 0.0
    bigram = model['bigram_counts']
    
    # 문장 앞뒤에 <s>와 </s>를 추가하고 띄어쓰기 기준으로 토큰화
    tokens = ['<s>'] + sentence.split() + ['</s>']
    
    # 단어가 2개 미만이면 바이그램을 만들 수 없으므로 -100 처리
    if len(tokens) < 2:
        return -100.0
        
    for i in range(len(tokens) - 1):
        w1 = tokens[i]
        w2 = tokens[i+1]
        
        # w1 다음에 w2가 나온 적이 있는지 확인
        if w1 in bigram and w2 in bigram[w1]:
            freq = bigram[w1][w2]
            # w1이 등장한 전체 횟수 구하기
            total_freq = sum(bigram[w1].values())
            
            # 확률 계산 (P = w1다음에 w2가 나온 횟수 / w1이 나온 전체 횟수)
            prob = freq / total_freq
            log_prob += math.log(prob)
        else:
            # 모델에 없는 단어 또는 바이그램인 경우
            log_prob += -100.0
            
    return log_prob

################################################################################
# 랜덤 문장 생성
# start_with : 생성할 문장의 시작 단어(들). 없으면 '<s>'로 초기화
def generate_sentence(model, start_with):
    # 시작 단어가 없는 경우
    if not start_with or start_with.strip() == "":
        tokens = ['<s>']
    # 시작 단어가 있는 경우
    else:
        tokens = ['<s>'] + start_with.strip().split()
        
    current_word = tokens[-1]
    max_length = 50 # 무한 루프 방지를 위한 최대 단어 수 제한
    
    # 문장 끝 기호(</s>)가 나오거나 최대 길이에 도달할 때까지 단어 생성
    while current_word != '</s>' and len(tokens) < max_length:
        next_word = get_next_word(model, current_word)
        tokens.append(next_word)
        current_word = next_word
        
    # 출력할 때 가상의 단어 <s>와 </s>는 제외하고 합침
    result_tokens = [w for w in tokens if w not in ('<s>', '</s>')]
    
    return ' '.join(result_tokens)

################################################################################
def load_model(model_file):
    with open(model_file, 'rb') as f:
        return pickle.load(f)

################################################################################
################################################################################
def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} model_file")
        sys.exit(1)
    
    model_file = sys.argv[1]
    model = load_model(model_file)
    
    print("2-gram 언어 모델 문장 생성기")
    
    while True:
        cmd = input("\n엔터 또는 문장 시작 단어(들) (q=종료): ")
        
        if cmd.lower() == 'q':
            print("프로그램을 종료합니다.")
            break
        else:
            print("\n<<<< 생성된 문장 >>>>")
            
            # 생성된 문장과 확률을 저장할 빈 리스트 생성
            results = []
            
            for _ in range(10):
                sentence = generate_sentence(model, cmd)
                log_prob = get_probability(model, sentence)
                # (로그 확률, 문장) 형태의 튜플로 리스트에 추가
                results.append((log_prob, sentence))
            
            # 리스트 정렬 (튜플의 첫 번째 요소인 log_prob을 기준으로 기본 내림차순 정렬됨)
            results.sort(key=lambda x: x[0], reverse=True)
            
            # 정렬된 결과를 순서대로 출력
            for i, (log_prob, sentence) in enumerate(results):
                print(f"문장{i+1} : {sentence} (로그 확률: {log_prob:.4f})")

################################################################################
if __name__ == "__main__":
    main()

################################################################################