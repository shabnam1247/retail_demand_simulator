// ==========================================================
// GLOBAL PLOTLY CONFIG
// ==========================================================

const plotConfig = {

    responsive: true,

    displayModeBar: false

};


// ==========================================================
// COMMON CHART LAYOUT
// ==========================================================

function chartLayout(title) {

    return {

        title: {

            text: title,

            font: {

                size: 15

            }

        },

        margin: {

            l: 45,

            r: 20,

            t: 55,

            b: 50

        },

        paper_bgcolor:
            "rgba(0,0,0,0)",

        plot_bgcolor:
            "rgba(0,0,0,0)",

        font: {

            family:
                "Inter, Arial",

            color:
                "#475569"

        },

        xaxis: {

            gridcolor:
                "#eef2f7",

            zeroline: false

        },

        yaxis: {

            gridcolor:
                "#eef2f7",

            zeroline: false

        }

    };

}


// ==========================================================
// DASHBOARD
// ==========================================================

async function loadDashboard() {

    try {

        const response =
            await fetch(
                "/api/summary"
            );

        const data =
            await response.json();


        // --------------------------------------------------
        // Revenue
        // --------------------------------------------------

        Plotly.newPlot(

            "revenueChart",

            [

                {

                    x:
                        data.monthly.map(
                            x => x.month
                        ),

                    y:
                        data.monthly.map(
                            x => x.revenue
                        ),

                    type:
                        "scatter",

                    mode:
                        "lines+markers",

                    line: {

                        width: 3

                    }

                }

            ],

            chartLayout(
                "Monthly Revenue"
            ),

            plotConfig

        );


        // --------------------------------------------------
        // Customer segments
        // --------------------------------------------------

        Plotly.newPlot(

            "segmentChart",

            [

                {

                    labels:
                        data.segments.map(
                            x => x.segment
                        ),

                    values:
                        data.segments.map(
                            x => x.customers
                        ),

                    type:
                        "pie",

                    hole:
                        0.55

                }

            ],

            chartLayout(
                "Customer Segments"
            ),

            plotConfig

        );


        // --------------------------------------------------
        // Products
        // --------------------------------------------------

        Plotly.newPlot(

            "productChart",

            [

                {

                    x:
                        data.products.map(
                            x => x.product_id
                        ),

                    y:
                        data.products.map(
                            x => x.revenue
                        ),

                    type:
                        "bar"

                }

            ],

            chartLayout(
                "Top Products by Revenue"
            ),

            plotConfig

        );


        // --------------------------------------------------
        // Weather
        // --------------------------------------------------

        Plotly.newPlot(

            "weatherChart",

            [

                {

                    x:
                        data.weather.map(
                            x => x.weather
                        ),

                    y:
                        data.weather.map(
                            x => x.units
                        ),

                    type:
                        "bar"

                }

            ],

            chartLayout(
                "Weather Impact"
            ),

            plotConfig

        );


        // --------------------------------------------------
        // Promotion
        // --------------------------------------------------

        Plotly.newPlot(

            "promotionChart",

            [

                {

                    x:
                        data.promotion.map(
                            x =>
                                x.promotion == 1
                                    ? "Promotion"
                                    : "No Promotion"
                        ),

                    y:
                        data.promotion.map(
                            x => x.avg_units
                        ),

                    type:
                        "bar"

                }

            ],

            chartLayout(
                "Promotion Effect"
            ),

            plotConfig

        );


        // --------------------------------------------------
        // Category
        // --------------------------------------------------

        Plotly.newPlot(

            "categoryChart",

            [

                {

                    labels:
                        data.category.map(
                            x => x.category
                        ),

                    values:
                        data.category.map(
                            x => x.revenue
                        ),

                    type:
                        "pie",

                    hole:
                        0.45

                }

            ],

            chartLayout(
                "Category Revenue"
            ),

            plotConfig

        );

    }

    catch (error) {

        console.error(
            "Dashboard error:",
            error
        );

    }

}


// ==========================================================
// INITIALIZE
// ==========================================================

document.addEventListener(
    "DOMContentLoaded",
    function () {

        if (
            document.getElementById(
                "revenueChart"
            )
        ) {

            loadDashboard();

        }

    }
);