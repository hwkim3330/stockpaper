# 요일별 주간 극단값 집중 현상의 재해석

[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Live-brightgreen)](https://hwkim3330.github.io/stockpaper/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Korean](https://img.shields.io/badge/language-Korean-blue.svg)](README.md)

## 📄 Overview

본 연구는 금융시장에서 주간 최고가와 최저가가 특정 요일에 집중되는 현상을 비판적으로 재검토합니다.

기존 연구(Lee, 2025, SSRN)는 주간 극단값이 특정 요일에 집중된다고 주장했지만, 본 연구는 이러한 현상이 **시장 구조적 요인**(변동성 계절성, 거래시간 비대칭성, 정보 흐름)으로 설명될 수 있음을 보입니다.

## 🔍 Main Findings

1. **구조적 요인 통제 시 패턴 소멸**
   - 변동성 정규화, 정보 점프 제거 후 G-검정 p-value > 0.9

2. **새로운 모형 제안: WCSV**
   - 요일조건부 확률적 변동성 모형
   - 기존 GARCH 모형보다 AIC/BIC 개선

3. **경제적 의미**
   - 패턴을 활용한 거래 전략은 비용 고려 시 수익 불가
   - 시장은 여전히 준효율적

## 📊 Results Summary

| Metric | Before Control | After Control |
|--------|---------------|---------------|
| G-statistic (Highs) | 12.34 (p<0.05) | 0.89 (p=0.93) |
| G-statistic (Lows) | 15.67 (p<0.01) | 1.12 (p=0.89) |
| Monday Volatility | 1.34% | 1.34% |
| Other Days Avg | 1.11% | 1.11% |

## 🚀 Quick Start

### View Online

Visit the [GitHub Pages site](https://hwkim3330.github.io/stockpaper/) to read the full paper.

### Local Setup

```bash
# Clone repository
git clone https://github.com/hwkim3330/stockpaper.git
cd stockpaper

# Install dependencies (optional, for code reproduction)
pip install -r requirements.txt

# Run analysis (if data available)
python scripts/wcsv_analysis.py
```

## 📚 Repository Structure

```
stockpaper/
├── paper.md              # 전체 논문 (Markdown)
├── index.md              # 홈페이지
├── about.md              # 연구 소개
├── _config.yml           # Jekyll 설정
├── scripts/              # 분석 코드 (향후 추가)
│   ├── wcsv_model.py
│   └── simulation.py
├── data/                 # 데이터 (향후 추가)
├── figures/              # 그림 (향후 추가)
└── README.md             # 이 파일
```

## 🛠️ Technology Stack

- **Markdown/Jekyll**: 문서 작성 및 GitHub Pages
- **Python 3.x**: 데이터 분석
  - NumPy, Pandas: 데이터 처리
  - SciPy: 통계 분석
  - Matplotlib, Seaborn: 시각화

## 📖 Citation

```bibtex
@article{kim2025weekday,
  title={요일별 주간 극단값 집중 현상의 재해석: 시장 미시구조와 시간 비균질성 분석},
  author={김현우},
  year={2025},
  month={10},
  url={https://hwkim3330.github.io/stockpaper/},
  note={GitHub repository: https://github.com/hwkim3330/stockpaper}
}
```

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

### How to contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This work is licensed under a [Creative Commons Attribution 4.0 International License](LICENSE).

- ✅ You are free to: Share, adapt
- ⚠️ You must: Give credit, indicate changes
- ❌ No additional restrictions

## 📧 Contact

- **Author**: 김현우 (Hyunwoo Kim)
- **Email**: hwkim3330@gmail.com
- **GitHub**: [@hwkim3330](https://github.com/hwkim3330)

## 🙏 Acknowledgments

- SSRN paper by Lee (2025) for initial inspiration
- ChatGPT for literature review assistance
- Claude Code for development support
- Open-source Python community

## 📅 Version History

- **v1.0.0** (2025-10-23): Initial release
  - Complete paper with WCSV model
  - GitHub Pages setup
  - Basic documentation

## 🔗 Links

- [Live Site](https://hwkim3330.github.io/stockpaper/)
- [Original SSRN Paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5283039)
- [Issues](https://github.com/hwkim3330/stockpaper/issues)
- [Pull Requests](https://github.com/hwkim3330/stockpaper/pulls)

---

**Note**: This is an academic research project. The findings are for educational purposes only and should not be considered investment advice.

**Keywords**: Stock market, Day-of-the-week effect, Extreme values, Volatility modeling, Market microstructure, Financial econometrics

---

Made with ❤️ using GitHub Pages and Jekyll