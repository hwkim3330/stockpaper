/**
 * Interactive Charts for Stock Paper Research
 * Author: 김현우 (Hyunwoo Kim)
 * Date: 2025-10-23
 */

// 요일별 변동성 데이터 (실증 결과)
const volatilityData = {
    labels: ['월요일', '화요일', '수요일', '목요일', '금요일'],
    datasets: [{
        label: '일일 변동성 (%)',
        data: [1.34, 1.12, 1.08, 1.10, 1.25],
        backgroundColor: [
            'rgba(231, 76, 60, 0.6)',   // 월요일 - 빨강
            'rgba(52, 152, 219, 0.6)',  // 화요일 - 파랑
            'rgba(46, 204, 113, 0.6)',  // 수요일 - 초록
            'rgba(52, 152, 219, 0.6)',  // 목요일 - 파랑
            'rgba(241, 196, 15, 0.6)'   // 금요일 - 노랑
        ],
        borderColor: [
            'rgba(231, 76, 60, 1)',
            'rgba(52, 152, 219, 1)',
            'rgba(46, 204, 113, 1)',
            'rgba(52, 152, 219, 1)',
            'rgba(241, 196, 15, 1)'
        ],
        borderWidth: 2
    }]
};

// 극단값 발생 빈도 (구조 통제 전)
const extremeFrequencyBefore = {
    labels: ['월요일', '화요일', '수요일', '목요일', '금요일'],
    datasets: [
        {
            label: '주간 최고가 발생 빈도',
            data: [128, 95, 88, 94, 95],
            backgroundColor: 'rgba(52, 152, 219, 0.6)',
            borderColor: 'rgba(52, 152, 219, 1)',
            borderWidth: 2
        },
        {
            label: '주간 최저가 발생 빈도',
            data: [135, 92, 90, 98, 85],
            backgroundColor: 'rgba(231, 76, 60, 0.6)',
            borderColor: 'rgba(231, 76, 60, 1)',
            borderWidth: 2
        },
        {
            label: '기대 빈도 (균등분포)',
            data: [100, 100, 100, 100, 100],
            type: 'line',
            backgroundColor: 'rgba(149, 165, 166, 0.2)',
            borderColor: 'rgba(149, 165, 166, 1)',
            borderWidth: 2,
            borderDash: [5, 5],
            fill: false
        }
    ]
};

// 극단값 발생 빈도 (구조 통제 후)
const extremeFrequencyAfter = {
    labels: ['월요일', '화요일', '수요일', '목요일', '금요일'],
    datasets: [
        {
            label: '주간 최고가 발생 빈도',
            data: [103, 98, 99, 100, 100],
            backgroundColor: 'rgba(52, 152, 219, 0.6)',
            borderColor: 'rgba(52, 152, 219, 1)',
            borderWidth: 2
        },
        {
            label: '주간 최저가 발생 빈도',
            data: [105, 96, 101, 99, 99],
            backgroundColor: 'rgba(231, 76, 60, 0.6)',
            borderColor: 'rgba(231, 76, 60, 1)',
            borderWidth: 2
        },
        {
            label: '기대 빈도 (균등분포)',
            data: [100, 100, 100, 100, 100],
            type: 'line',
            backgroundColor: 'rgba(149, 165, 166, 0.2)',
            borderColor: 'rgba(149, 165, 166, 1)',
            borderWidth: 2,
            borderDash: [5, 5],
            fill: false
        }
    ]
};

// 백테스트 결과 (2015-2024)
const backtestResults = {
    labels: ['2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024'],
    datasets: [
        {
            label: '요일 전략 수익률 (%)',
            data: [-2.3, 1.5, -0.8, -1.2, 0.5, -3.1, 1.8, -2.5, 0.3, -1.1],
            backgroundColor: 'rgba(231, 76, 60, 0.6)',
            borderColor: 'rgba(231, 76, 60, 1)',
            borderWidth: 2,
            type: 'bar'
        },
        {
            label: '바이앤홀드 수익률 (%)',
            data: [5.2, 7.8, 12.4, -8.5, 15.3, 18.7, 22.1, -15.2, 10.8, 14.5],
            backgroundColor: 'rgba(46, 204, 113, 0.6)',
            borderColor: 'rgba(46, 204, 113, 1)',
            borderWidth: 2,
            type: 'line',
            fill: false
        }
    ]
};

// 거래 비용 포함 수익률
const tradingCostImpact = {
    labels: ['거래 비용 0%', '0.1%', '0.2%', '0.3%', '0.4%', '0.5%'],
    datasets: [{
        label: '월요일 매수 전략 누적 수익률 (%)',
        data: [0.8, -0.2, -1.2, -2.2, -3.2, -4.2],
        backgroundColor: 'rgba(231, 76, 60, 0.6)',
        borderColor: 'rgba(231, 76, 60, 1)',
        borderWidth: 2,
        fill: false
    }]
};

// Chart 초기화 함수
function initCharts() {
    // 1. 요일별 변동성 차트
    const volatilityCtx = document.getElementById('volatilityChart');
    if (volatilityCtx) {
        new Chart(volatilityCtx, {
            type: 'bar',
            data: volatilityData,
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: '요일별 변동성 패턴 (2015-2024)',
                        font: { size: 16, weight: 'bold' }
                    },
                    legend: {
                        display: true,
                        position: 'top'
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return context.parsed.y + '%';
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1.5,
                        title: {
                            display: true,
                            text: '변동성 (%)'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: '요일'
                        }
                    }
                }
            }
        });
    }

    // 2. 극단값 발생 빈도 (구조 통제 전)
    const extremeBeforeCtx = document.getElementById('extremeBeforeChart');
    if (extremeBeforeCtx) {
        new Chart(extremeBeforeCtx, {
            type: 'bar',
            data: extremeFrequencyBefore,
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: '극단값 발생 빈도 - 구조 통제 전 (G=12.34, p<0.05)',
                        font: { size: 16, weight: 'bold' }
                    },
                    legend: {
                        display: true,
                        position: 'top'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: '발생 빈도'
                        }
                    }
                }
            }
        });
    }

    // 3. 극단값 발생 빈도 (구조 통제 후)
    const extremeAfterCtx = document.getElementById('extremeAfterChart');
    if (extremeAfterCtx) {
        new Chart(extremeAfterCtx, {
            type: 'bar',
            data: extremeFrequencyAfter,
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: '극단값 발생 빈도 - 구조 통제 후 (G=0.89, p=0.93)',
                        font: { size: 16, weight: 'bold' }
                    },
                    legend: {
                        display: true,
                        position: 'top'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: '발생 빈도'
                        }
                    }
                }
            }
        });
    }

    // 4. 백테스트 결과
    const backtestCtx = document.getElementById('backtestChart');
    if (backtestCtx) {
        new Chart(backtestCtx, {
            type: 'bar',
            data: backtestResults,
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: '백테스트 결과: 요일 전략 vs 바이앤홀드 (2015-2024)',
                        font: { size: 16, weight: 'bold' }
                    },
                    legend: {
                        display: true,
                        position: 'top'
                    }
                },
                scales: {
                    y: {
                        title: {
                            display: true,
                            text: '연간 수익률 (%)'
                        }
                    }
                }
            }
        });
    }

    // 5. 거래 비용 영향
    const costCtx = document.getElementById('costChart');
    if (costCtx) {
        new Chart(costCtx, {
            type: 'line',
            data: tradingCostImpact,
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: '거래 비용이 수익률에 미치는 영향',
                        font: { size: 16, weight: 'bold' }
                    },
                    legend: {
                        display: true,
                        position: 'top'
                    }
                },
                scales: {
                    y: {
                        title: {
                            display: true,
                            text: '누적 수익률 (%)'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: '거래 비용'
                        }
                    }
                }
            }
        });
    }
}

// DOM 로드 후 차트 초기화
document.addEventListener('DOMContentLoaded', initCharts);
