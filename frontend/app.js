const API_BASE_URL = "/api";

const citySelect = document.getElementById("citySelect");
const citySelectMetrics = document.getElementById("citySelectMetrics");
const modelSelect = document.getElementById("modelSelect");
const horizonSelect = document.getElementById("horizonSelect");
const forecastButton = document.getElementById("forecastButton");
const updateButton = document.getElementById("updateButton");
const updateStatus = document.getElementById("updateStatus");
const automationStatus = document.getElementById("automationStatus");
const comparisonStatus = document.getElementById("comparisonStatus");
const forecastStatus = document.getElementById("forecastStatus");
const forecastTableBody = document.querySelector("#forecastTable tbody");
const metricsButton = document.getElementById("metricsButton");
const metricsStatus = document.getElementById("metricsStatus");
const metricsTableBody = document.querySelector("#metricsTable tbody");

let forecastChart = null;

/**
 * Прочитать JSON-ответ API и обработать ошибочный статус.
 *
 * @param {Response} response Ответ сервера.
 * @returns {Promise<object>} Данные ответа.
 */
async function getJson(response) {
    let data;
    try {
        data = await response.json();
    } catch {
        throw new Error(`Сервер вернул некорректный ответ (${response.status}).`);
    }

    if (!response.ok) {
        const detail = Array.isArray(data.detail)
            ? data.detail.map((item) => item.msg).join("; ")
            : data.detail;
        throw new Error(detail || `Ошибка запроса (${response.status}).`);
    }
    return data;
}

/**
 * Показать сообщение в пустой таблице.
 *
 * @param {HTMLTableSectionElement} tableBody Тело таблицы.
 * @param {number} columns Число колонок таблицы.
 * @param {string} message Текст сообщения.
 */
function renderEmptyTable(tableBody, columns, message) {
    tableBody.innerHTML =
        `<tr class="empty-row"><td colspan="${columns}">${message}</td></tr>`;
}

/**
 * Загрузить сравнение прогнозов с фактической погодой.
 *
 * @param {string} city Идентификатор города.
 * @param {string} model Название модели.
 * @returns {Promise<object>} Точки сравнения и метрики.
 */
async function loadComparison(city, model) {
    const response = await fetch(
        `${API_BASE_URL}/models/comparison?city=${city}&model_name=${model}`
    );
    return getJson(response);
}

/**
 * Отобразить постфактум метрики сохранённых прогнозов.
 *
 * @param {object|null} metrics Метрики сравнения или null.
 */
function renderComparisonMetrics(metrics) {
    if (!metrics) {
        comparisonStatus.textContent =
            "Постфактум проверка появится после наступления дат прогноза.";
        return;
    }

    comparisonStatus.textContent =
        `Постфактум: MAE ${metrics.mae}, RMSE ${metrics.rmse}, ` +
        `проверено точек: ${metrics.compared_rows}.`;
}

/**
 * Загрузить и показать статус автоматического обновления.
 */
async function loadAutomationStatus() {
    try {
        const response = await fetch(`${API_BASE_URL}/automation/status`);
        const data = await getJson(response);

        if (!data.finished_at) {
            automationStatus.textContent =
                "Автоматическое обновление ещё не выполнялось.";
            return;
        }

        const finishedAt = new Date(data.finished_at).toLocaleString("ru-RU");
        automationStatus.textContent =
            `Последнее автоматическое обновление: ${finishedAt}.`;
    } catch (error) {
        automationStatus.textContent =
            `Статус автообновления недоступен: ${error.message}`;
    }
}

// ── Tab switching ──
document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
        document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
        document.querySelectorAll(".tab-content").forEach((c) => c.classList.remove("active"));
        tab.classList.add("active");
        document.getElementById(`tab-${tab.dataset.tab}`).classList.add("active");
    });
});

// Keep city selects in sync
citySelect.addEventListener("change", () => { citySelectMetrics.value = citySelect.value; });
citySelectMetrics.addEventListener("change", () => { citySelect.value = citySelectMetrics.value; });

updateButton.addEventListener("click", async () => {
    const city = citySelect.value;
    const model = modelSelect.value;

    updateButton.textContent = "Обновление и обучение...";
    updateButton.disabled = true;
    updateStatus.className = "status-message";
    updateStatus.textContent = "Загрузка погодных данных...";

    try {
        const response = await fetch(
            `${API_BASE_URL}/models/update-and-train?city=${city}&model_name=${model}`,
            { method: "POST" }
        );
        const data = await getJson(response);
        updateStatus.classList.add("success");
        updateStatus.textContent =
            `Готово: добавлено ${data.data.added_rows} записей, ` +
            `MAE ${data.metrics.mae}, RMSE ${data.metrics.rmse}.`;
        await loadAutomationStatus();
    } catch (error) {
        updateStatus.classList.add("error");
        updateStatus.textContent = error.message;
    } finally {
        updateButton.textContent = "Обновить данные / переобучить";
        updateButton.disabled = false;
    }
});

// ── Forecast ──
forecastButton.addEventListener("click", async () => {
    const city = citySelect.value;
    const model = modelSelect.value;

    forecastButton.textContent = "Загрузка…";
    forecastButton.disabled = true;
    forecastStatus.className = "status-message";
    forecastStatus.textContent = "Построение прогноза...";

    try {
        const [forecastRes, actualRes] = await Promise.all([
            fetch(`${API_BASE_URL}/models/forecast?city=${city}&model_name=${model}&horizon=${horizonSelect.value}`),
            fetch(`${API_BASE_URL}/weather/dataset?city=${city}`)
        ]);

        const data = await getJson(forecastRes);
        const actualData = (await getJson(actualRes)).slice(-14);
        const comparison = await loadComparison(city, model);
        renderComparisonMetrics(comparison.metrics);

        const actualByDate = new Map(
            actualData.map((item) => [item.date, item.temp_mean])
        );
        comparison.points.forEach((item) => {
            actualByDate.set(item.date, item.actual);
        });

        const forecastByDate = new Map(
            comparison.points.map((item) => [item.date, item.prediction])
        );
        data.forecast.forEach((item) => {
            forecastByDate.set(item.date, item.predicted_temp_mean);
        });

        forecastTableBody.innerHTML = "";
        data.forecast.forEach((item) => {
            const row = document.createElement("tr");
            row.innerHTML = `<td>${item.date}</td><td>${item.predicted_temp_mean} °C</td>`;
            forecastTableBody.appendChild(row);
        });

        const labels = Array.from(
            new Set([...actualByDate.keys(), ...forecastByDate.keys()])
        ).sort();
        const actualValues = labels.map(
            (date) => actualByDate.get(date) ?? null
        );
        const forecastValues = labels.map(
            (date) => forecastByDate.get(date) ?? null
        );
        const allValues = [...actualValues, ...forecastValues].filter(
            (value) => value !== null
        );

        if (forecastChart) forecastChart.destroy();

        forecastChart = new Chart(document.getElementById("forecastChart"), {
            type: "line",
            data: {
                labels,
                datasets: [
                    {
                        label: "Факт",
                        data: actualValues,
                        borderWidth: 2,
                        borderColor: "#3b82f6",
                        backgroundColor: "rgba(59,130,246,0.08)",
                        fill: true,
                        tension: 0.3,
                        pointRadius: 3,
                        pointHoverRadius: 5,
                    },
                    {
                        label: `Прогноз: ${data.model}`,
                        data: forecastValues,
                        borderWidth: 2,
                        borderColor: "#ef4444",
                        backgroundColor: "rgba(239,68,68,0.07)",
                        fill: true,
                        tension: 0.3,
                        borderDash: [5, 3],
                        pointRadius: 3,
                        pointHoverRadius: 5,
                    },
                ],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: { label: (ctx) => ` ${ctx.parsed.y}°C` },
                        backgroundColor: "#1e2333",
                        borderColor: "#2a3045",
                        borderWidth: 1,
                        titleColor: "#7c86a1",
                        bodyColor: "#e8eaf0",
                    },
                },
                scales: {
                    x: {
                        grid: { color: "rgba(255,255,255,0.04)" },
                        ticks: { color: "#7c86a1", font: { family: "JetBrains Mono", size: 11 }, maxRotation: 30 },
                    },
                    y: {
                        suggestedMin: Math.min(...allValues) - 5,
                        suggestedMax: Math.max(...allValues) + 5,
                        grid: { color: "rgba(255,255,255,0.04)" },
                        ticks: { color: "#7c86a1", font: { family: "JetBrains Mono", size: 11 }, callback: (v) => `${v}°C` },
                    },
                },
            },
        });
        forecastStatus.classList.add("success");
        forecastStatus.textContent =
            `Прогноз модели ${data.model} построен на ${data.horizon} дней.`;
    } catch (error) {
        forecastStatus.classList.add("error");
        forecastStatus.textContent = error.message;
        renderEmptyTable(
            forecastTableBody,
            2,
            "Не удалось получить прогноз."
        );
    } finally {
        forecastButton.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg> Спрогнозировать`;
        forecastButton.disabled = false;
    }
});

// ── Metrics ──
metricsButton.addEventListener("click", async () => {
    const city = citySelectMetrics.value;

    metricsButton.textContent = "Загрузка…";
    metricsButton.disabled = true;
    metricsStatus.className = "status-message";
    metricsStatus.textContent = "Загрузка метрик...";

    try {
        const response = await fetch(`${API_BASE_URL}/models/metrics?city=${city}`);
        const data = await getJson(response);

        metricsTableBody.innerHTML = "";

        const items = Object.entries(data.metrics).filter(([, m]) => m !== null);
        const badge = document.getElementById("bestModel");

        if (items.length === 0) {
            badge.textContent = "";
            metricsStatus.textContent =
                "Для выбранного города ещё нет обученных моделей.";
            renderEmptyTable(
                metricsTableBody,
                3,
                "Сначала обновите данные и обучите модель."
            );
            return;
        }

        const best = items.reduce((b, c) => (c[1].mae < b[1].mae ? c : b));

        items.forEach(([name, metrics]) => {
            const row = document.createElement("tr");
            if (name === best[0]) row.classList.add("best-row");
            row.innerHTML = `<td>${name}</td><td>${metrics.mae}</td><td>${metrics.rmse}</td>`;
            metricsTableBody.appendChild(row);
        });

        badge.textContent = `${best[0]} · MAE ${best[1].mae}`;
        metricsStatus.classList.add("success");
        metricsStatus.textContent = `Загружено моделей: ${items.length}.`;
    } catch (error) {
        metricsStatus.classList.add("error");
        metricsStatus.textContent = error.message;
        document.getElementById("bestModel").textContent = "";
        renderEmptyTable(metricsTableBody, 3, "Не удалось загрузить метрики.");
    } finally {
        metricsButton.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18M9 21V9"/></svg> Сравнить модели`;
        metricsButton.disabled = false;
    }
});

// ── Init ──
async function loadCities() {
    const res = await fetch(`${API_BASE_URL}/cities`);
    const cities = await getJson(res);
    [citySelect, citySelectMetrics].forEach((sel) => {
        sel.innerHTML = "";
        Object.entries(cities).forEach(([id, c]) => {
            const o = document.createElement("option");
            o.value = id; o.textContent = c.name;
            sel.appendChild(o);
        });
    });
}

async function loadModels() {
    const res = await fetch(`${API_BASE_URL}/models/list`);
    const data = await getJson(res);
    modelSelect.innerHTML = "";
    data.models.forEach((name) => {
        const o = document.createElement("option");
        o.value = name; o.textContent = name;
        modelSelect.appendChild(o);
    });
}

Promise.all([loadCities(), loadModels()]).catch((error) => {
    forecastStatus.className = "status-message error";
    forecastStatus.textContent =
        `Не удалось загрузить настройки приложения: ${error.message}`;
    forecastButton.disabled = true;
    updateButton.disabled = true;
    metricsButton.disabled = true;
});
loadAutomationStatus();
