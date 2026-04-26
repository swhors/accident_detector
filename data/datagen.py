import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def gen_data(output_file_namepath: str):
    # 1. 초기 설정 및 현장별 상세 스펙 반영
    np.random.seed(42)
    start_date = datetime(2021, 1, 1)
    total_days = 365 * 3
    accident_records = []

    sites_config = {
        'A': {
            'name': '아파트(10개동/17층)', 
            'perm_workers': 300, 'temp_workers': 200, 
            'avg_age': 40, 'edu': 30, 'night_shift': True,
            'base_prob': 0.025  # 가장 높은 사고 빈도
        },
        'B': {
            'name': '오피스텔(2개동/20층)', 
            'perm_workers': 200, 'temp_workers': 0, 
            'avg_age': 35, 'edu': 5, 'night_shift': False,
            'base_prob': 0.018  # 중간 사고 빈도
        },
        'C': {
            'name': '초고층사무용(2개동/70층)', 
            'perm_workers': 250, 'temp_workers': 100, 
            'avg_age': 45, 'edu': 15, 'night_shift': True,
            'base_prob': 0.012  # 가장 낮은 사고 빈도 (안전관리 엄격 가정)
        }
    }

    # 2. 데이터 생성 루프
    for i in range(total_days):
        curr_date = start_date + timedelta(days=i)
        month = curr_date.month
        season = '겨울'
        if 3 <= month <= 5: season = '봄'
        elif 6 <= month <= 8: season = '여름'
        elif 9 <= month <= 11: season = '가을'
    
        # 기상 조건 (여름철 비 확률)
        is_rainy = np.random.choice([True, False], p=[0.3, 0.7] if season == '여름' else [0.1, 0.9])

        for sid, info in sites_config.items():
            # [확률 설계]
            prob = info['base_prob']
            
            # 조건: A, C는 여름(비)에 위험도 상승
            if sid in ['A', 'C'] and season == '여름' and is_rainy:
                prob *= 2.5
        
            # 조건: B는 요청하신 계절 비중 적용 (봄 30% 등)
            if sid == 'B':
                b_season_weight = {'봄': 1.5, '여름': 1.25, '가을': 1.0, '겨울': 1.25}
                prob *= b_season_weight[season]
    
            # 사고 발생 시뮬레이션
            if np.random.rand() < prob:
                # 근로자 유형 결정 (비상시 근로자가 있는 현장만)
                worker_type = '상시'
                if info['temp_workers'] > 0:
                    total_w = info['perm_workers'] + info['temp_workers']
                    # 비상시(임시) 근로자가 숙련도 부족으로 사고 확률이 1.5배 높다고 가정
                    temp_p = (info['temp_workers'] * 1.5) / total_w
                    worker_type = '비상시' if np.random.rand() < temp_p else '상시'
    
                # 성별 결정 (A현장: 남녀 사고 건수 동일 로직)
                if sid == 'A':
                    gender = '여성' if np.random.rand() < 0.5 else '남성'
                elif sid == 'B':
                    gender = '남성'
                else:
                    gender = '여성' if np.random.rand() < 0.2 else '남성'
    
                # 사고 종류 (C현장은 70층이므로 추락사 비중 증가)
                acc_types = ['낙상', '화상', '단순 골절', '추락사']
                p_dist = [0.5, 0.2, 0.2, 0.1] if sid != 'C' else [0.3, 0.1, 0.2, 0.4]
    
                accident_records.append({
                    'date': curr_date.strftime('%Y-%m-%d'),
                    'site': sid,
                    'site_name': info['name'],
                    'worker_type': worker_type,
                    'gender': gender,
                    'age': int(np.random.normal(info['avg_age'], 6)),
                    'season': season,
                    'is_rainy': is_rainy,
                    'working_shift': '주간' if not info['night_shift'] else np.random.choice(['주간', '야간'], p=[0.7, 0.3]),
                    'accident_type': np.random.choice(acc_types, p=p_dist),
                    'edu_time': info['edu']
                })
    
    df_final = pd.DataFrame(accident_records)
    print(f"총 {len(df_final)}건의 데이터 생성 완료.")
    print(df_final.sample(10)) # 무작위 10개 확인

    if output_file_namepath is not None and len(output_file_namepath) > 0:
        df_final.to_csv('construction_accidents_3y.csv', index=False, encoding='utf-8-sig')
    return df_final

