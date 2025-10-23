# Analysis Scripts

이 디렉토리는 논문 분석에 사용된 Python 스크립트를 포함합니다.

## 파일 설명

### 1. `wcsv_model.py`
요일조건부 확률적 변동성(WCSV) 모형 구현

**주요 기능:**
- WCSV 모형 추정 (최대우도법)
- 요일별 파라미터 추정
- 조건부 변동성 예측
- AIC/BIC 계산

**사용 예시:**
```python
from wcsv_model import WCSVModel

# 모형 생성
model = WCSVModel()

# 데이터 준비 (returns: 수익률, weekday: 요일 인덱스 0-4)
model.fit(returns, weekday)

# 결과 출력
model.summary()
params_df = model.get_parameters()
```

### 2. `g_test.py`
G-검정 (Likelihood Ratio Test) 구현

**주요 기능:**
- G-검정 수행
- 주간 극단값의 요일별 분포 분석
- 변동성 조정 분석
- 결과 시각화

**사용 예시:**
```python
from g_test import analyze_weekly_extremes, g_test

# 주간 극단값 분석
results = analyze_weekly_extremes(prices, dates)

# G-검정 수행
high_test = g_test(results['high_counts'])
print(f"G-statistic: {high_test['G']:.2f}")
print(f"p-value: {high_test['p_value']:.4f}")
```

### 3. `simulation.py`
몬테카를로 시뮬레이션

**주요 기능:**
- WCSV 모형 시뮬레이션
- 표준 GARCH 모형 시뮬레이션
- 모형 비교 (WCSV vs GARCH)
- 결과 시각화

**사용 예시:**
```python
from simulation import compare_models

# 시뮬레이션 실행 (1000회, 각 500주)
results = compare_models(n_simulations=1000, n_weeks=500)

# 결과 확인
print(results['wcsv_high_props'].mean(axis=0))  # WCSV 평균 비율
print(results['garch_high_props'].mean(axis=0))  # GARCH 평균 비율
```

## 실행 방법

### 1. 의존성 설치
```bash
pip install -r requirements.txt
```

### 2. 개별 스크립트 실행
```bash
# WCSV 모형 예제
python scripts/wcsv_model.py

# G-검정 예제
python scripts/g_test.py

# 시뮬레이션 예제
python scripts/simulation.py
```

### 3. 패키지로 사용
```python
import sys
sys.path.append('.')

from scripts import WCSVModel, g_test, compare_models

# 분석 수행
# ...
```

## 출력 결과

스크립트 실행 시 다음 디렉토리에 결과가 저장됩니다:

- `figures/`: 그래프 및 시각화 결과 (PNG 형식)
- `data/`: 중간 데이터 파일 (향후 추가 예정)

## 참고사항

- 모든 스크립트는 독립적으로 실행 가능합니다
- 예제 데이터는 합성 데이터(synthetic data)를 사용합니다
- 실제 데이터를 사용하려면 `data/` 디렉토리에 데이터 파일을 추가하세요

## 라이선스

CC BY 4.0 - 자유롭게 사용 가능하며, 출처를 명시해주세요.

## 문의

- 저자: 김현우 (Hyunwoo Kim)
- 이메일: hwkim3330@gmail.com
