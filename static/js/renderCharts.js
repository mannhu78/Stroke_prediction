let myChart; // Biến lưu biểu đồ hiện tại
let initialChartDrawn = false; // Biến kiểm soát việc render chart ban đầu

document.addEventListener("DOMContentLoaded", function () {
    const chartContainer = document.getElementById("chart-container");
    const loadMoreBtn = document.getElementById("load-more");
    const showChartBtn = document.getElementById("show-chart");
    const rows = document.querySelectorAll(".result-row");
    let currentVisible = 10;
    const batchSize = 10;

    // Bắt sự kiện các nút biểu đồ
    document.querySelectorAll('.chart-btn').forEach(btn => {
        btn.addEventListener('click', function () {
            const chartType = this.dataset.chart;
            chartContainer.style.display = "block";

            if (myChart) {
                myChart.destroy();
            }

            switch (chartType) {
                case 'strokeRate':
                    renderStrokeRateChart();
                    break;
                case 'ageByGender':
                    renderAgeByGenderChart();
                    break;
                case 'workVsStroke':
                    renderWorkVsStrokeChart();
                    break;
                case 'correlationMatrix':
                    renderCorrelationMatrix();
                    break;
                case 'heartVsStroke':
                    renderHeartVsStrokeChart();
                    break;
                case 'smokeVsStroke':
                    renderSmokeVsStrokeChart();
                    break;
                default:
                    break;
            }
        });
    });

    // Nút "Xem thêm"
    loadMoreBtn?.addEventListener("click", function () {
        let revealed = 0;
        for (let i = currentVisible; i < rows.length && revealed < batchSize; i++) {
            rows[i].classList.remove("hidden");
            rows[i].classList.add("fade-in");
            revealed++;
        }
        currentVisible += batchSize;

        if (currentVisible >= rows.length) {
            loadMoreBtn.style.display = "none";
        }
    });

    // Nút "Xem biểu đồ tổng"
    showChartBtn?.addEventListener("click", function () {
        chartContainer.style.display = "block";
        showChartBtn.disabled = true;
        showChartBtn.style.display = "none";

        if (!initialChartDrawn) {
            renderMainChart();
            initialChartDrawn = true;
        }
    });
});

// ======================================
// Các hàm vẽ biểu đồ cụ thể
// ======================================

// 1. Biểu đồ tỉ lệ đột quỵ
function renderStrokeRateChart() {
    const ctx = document.getElementById('prediction-chart').getContext('2d');
    const strokeYes = {{ results|selectattr('nguy_co_dot_quy', 'equalto', 1)|list|length }};
    const strokeNo = {{ results|selectattr('nguy_co_dot_quy', 'equalto', 0)|list|length }};

    myChart = new Chart(ctx, {
        type: 'pie',
        data: {
            labels: ['Có đột quỵ', 'Không đột quỵ'],
            datasets: [{
                data: [strokeYes, strokeNo],
                backgroundColor: ['#ff6384', '#36a2eb']
            }]
        },
        options: { responsive: true }
    });
}

// 2. Biểu đồ tuổi theo giới tính
function renderAgeByGenderChart() {
    const ctx = document.getElementById('prediction-chart').getContext('2d');
    const males = {{ results|selectattr('gioi_tinh', 'equalto', 0)|map(attribute='tuoi')|list }};
    const females = {{ results|selectattr('gioi_tinh', 'equalto', 1)|map(attribute='tuoi')|list }};

    myChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Nam', 'Nữ'],
            datasets: [{
                label: 'Tuổi trung bình',
                data: [
                    males.reduce((a, b) => a + b, 0) / (males.length || 1),
                    females.reduce((a, b) => a + b, 0) / (females.length || 1)
                ],
                backgroundColor: ['#4bc0c0', '#ff9f40']
            }]
        },
        options: {
            responsive: true,
            scales: { y: { beginAtZero: true } }
        }
    });
}

// 3. Nghề nghiệp & đột quỵ
function renderWorkVsStrokeChart() {
    const ctx = document.getElementById('prediction-chart').getContext('2d');
    const labels = ['Văn phòng', 'Công nhân', 'Tự kinh doanh', 'Trẻ em', 'Khác'];
    const data = [0, 0, 0, 0, 0];

    {% for row in results %}
        {% if row.nguy_co_dot_quy == 1 %}
            data[{{ row.cong_viec }}]++;
        {% endif %}
    {% endfor %}

    myChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Số ca đột quỵ',
                data: data,
                backgroundColor: '#9966ff'
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            scales: { x: { beginAtZero: true } }
        }
    });
}

// 4. Ma trận tương quan BMI vs Mức đường huyết
function renderCorrelationMatrix() {
    const ctx = document.getElementById('prediction-chart').getContext('2d');

    const dataPoints = [
        {% for row in results %}
            { x: {{ row.bmi }}, y: {{ row.muc_duong_huyet }} },
        {% endfor %}
    ];

    myChart = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'BMI vs Mức đường huyết',
                data: dataPoints,
                backgroundColor: '#ff6384'
            }]
        },
        options: {
            responsive: true,
            scales: {
                x: { title: { display: true, text: 'BMI' }},
                y: { title: { display: true, text: 'Mức đường huyết' }}
            }
        }
    });
}

// 5. Tỉ lệ đột quỵ theo bệnh tim
function renderHeartVsStrokeChart() {
    const ctx = document.getElementById('prediction-chart').getContext('2d');
    const heartYes = {{ results|selectattr('benh_tim', 'equalto', 1)|selectattr('nguy_co_dot_quy', 'equalto', 1)|list|length }};
    const heartNo = {{ results|selectattr('benh_tim', 'equalto', 0)|selectattr('nguy_co_dot_quy', 'equalto', 1)|list|length }};

    myChart = new Chart(ctx, {
        type: 'pie',
        data: {
            labels: ['Có bệnh tim', 'Không bệnh tim'],
            datasets: [{
                data: [heartYes, heartNo],
                backgroundColor: ['#ffcd56', '#36a2eb']
            }]
        },
        options: { responsive: true }
    });
}

// 6. Tỉ lệ đột quỵ theo tình trạng hút thuốc
function renderSmokeVsStrokeChart() {
    const ctx = document.getElementById('prediction-chart').getContext('2d');
    const labels = ['Không hút', 'Đang hút', 'Đã từng hút', 'Không xác định'];
    const data = [0, 0, 0, 0];

    {% for row in results %}
        {% if row.nguy_co_dot_quy == 1 %}
            data[{{ row.hut_thuoc }}]++;
        {% endif %}
    {% endfor %}

    myChart = new Chart(ctx, {
        type: 'pie',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: ['#36a2eb', '#ff6384', '#ff9f40', '#4bc0c0']
            }]
        },
        options: { responsive: true }
    });
}

// 7. Biểu đồ tổng ban đầu: phần trăm nguy cơ đột quỵ
function renderMainChart() {
    const ctx = document.getElementById('prediction-chart').getContext('2d');

    const labels = [{% for item in results %}'{{ item.ma_benh_nhan }}'{% if not loop.last %}, {% endif %}{% endfor %}];
    const dataPoints = [{% for item in results %}{{ item.phan_tram_du_doan }}{% if not loop.last %}, {% endif %}{% endfor %}];

    if (labels.length > 0 && dataPoints.length > 0) {
        myChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Nguy cơ đột quỵ (%)',
                    data: dataPoints,
                    backgroundColor: 'rgba(255, 99, 132, 0.2)',
                    borderColor: 'rgba(255, 99, 132, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                scales: { y: { beginAtZero: true } }
            }
        });
    }
}
