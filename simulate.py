from pathlib import Path
import numpy as np
import pandas as pd


# ==========================================================
# CONFIGURATION
# ==========================================================

SEED = 42

rng = np.random.default_rng(SEED)

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"

DATA_DIR.mkdir(exist_ok=True)


# ==========================================================
# HELPER
# ==========================================================

def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


# ==========================================================
# CUSTOMERS
# ==========================================================

def generate_customers(n=800):

    customer_ids = [
        f"C{i:04d}"
        for i in range(1, n + 1)
    ]

    customers = pd.DataFrame({

        "customer_id": customer_ids,

        "age": rng.integers(
            18,
            65,
            n
        ),

        "income_band": rng.choice(
            [
                "Low",
                "Medium",
                "High"
            ],
            n,
            p=[
                0.25,
                0.55,
                0.20
            ]
        ),

        "loyalty_tier": rng.choice(
            [
                "Bronze",
                "Silver",
                "Gold",
                "Platinum"
            ],
            n,
            p=[
                0.40,
                0.30,
                0.22,
                0.08
            ]
        ),

        "city": rng.choice(
            [
                "Kochi",
                "Kozhikode",
                "Malappuram",
                "Thrissur",
                "Kannur"
            ],
            n
        )
    })

    income_multiplier = {
        "Low": 0.85,
        "Medium": 1.00,
        "High": 1.35
    }

    loyalty_multiplier = {
        "Bronze": 0.90,
        "Silver": 1.00,
        "Gold": 1.15,
        "Platinum": 1.30
    }

    customers["customer_value_factor"] = (
        customers["income_band"]
        .map(income_multiplier)
        *
        customers["loyalty_tier"]
        .map(loyalty_multiplier)
    )

    customers.to_csv(
        DATA_DIR / "customers.csv",
        index=False
    )

    return customers


# ==========================================================
# PRODUCTS
# ==========================================================

def generate_products():

    products = [

        ["P001", "Samsung", "Galaxy A15",
         "Smartphone", "Budget", 22000, 0.28, 1.30],

        ["P002", "Samsung", "Galaxy S24 FE",
         "Smartphone", "Premium", 59000, 0.20, 0.90],

        ["P003", "Apple", "iPhone 15",
         "Smartphone", "Premium", 69900, 0.18, 0.85],

        ["P004", "Apple", "iPhone 15 Pro",
         "Smartphone", "Premium", 129900, 0.16, 0.55],

        ["P005", "OnePlus", "Nord CE 4",
         "Smartphone", "Midrange", 24999, 0.22, 1.20],

        ["P006", "OnePlus", "12R",
         "Smartphone", "Premium", 39999, 0.20, 1.00],

        ["P007", "Xiaomi", "Redmi Note 14",
         "Smartphone", "Budget", 17999, 0.26, 1.40],

        ["P008", "Realme", "12 Pro",
         "Smartphone", "Midrange", 24999, 0.25, 1.15],

        ["P009", "Vivo", "V30",
         "Smartphone", "Midrange", 33999, 0.22, 1.00],

        ["P010", "Oppo", "Reno 12",
         "Smartphone", "Midrange", 32999, 0.22, 0.95],

        ["P011", "boAt", "Airdopes 141",
         "Audio", "Budget", 1299, 0.35, 2.30],

        ["P012", "Sony", "WH-CH520",
         "Audio", "Midrange", 4490, 0.25, 1.40],

        ["P013", "JBL", "Tune 770NC",
         "Audio", "Midrange", 6999, 0.23, 1.20],

        ["P014", "Samsung", "Galaxy Buds FE",
         "Audio", "Midrange", 7999, 0.20, 1.10],

        ["P015", "Apple", "AirPods 3",
         "Audio", "Premium", 19900, 0.18, 0.70],

        ["P016", "Samsung", "25W Charger",
         "Accessories", "Budget", 1499, 0.32, 2.50],

        ["P017", "Anker", "PowerCore 20K",
         "Accessories", "Midrange", 3999, 0.27, 1.70],

        ["P018", "Spigen", "Phone Case",
         "Accessories", "Budget", 1499, 0.38, 2.80],

        ["P019", "Apple", "20W Adapter",
         "Accessories", "Midrange", 1900, 0.28, 1.90],

        ["P020", "Samsung", "Galaxy Tab A9",
         "Tablet", "Midrange", 17999, 0.22, 0.70]
    ]

    products = pd.DataFrame(
        products,
        columns=[
            "product_id",
            "brand",
            "product_name",
            "category",
            "price_band",
            "base_price",
            "margin_rate",
            "demand_index"
        ]
    )

    products.to_csv(
        DATA_DIR / "products.csv",
        index=False
    )

    return products


# ==========================================================
# STORES
# ==========================================================

def generate_stores():

    stores = pd.DataFrame({

        "store_id": [
            "S001",
            "S002",
            "S003",
            "S004",
            "S005"
        ],

        "store_name": [
            "Kochi Central",
            "Kozhikode Main",
            "Malappuram Town",
            "Thrissur Centre",
            "Kannur Market"
        ],

        "city": [
            "Kochi",
            "Kozhikode",
            "Malappuram",
            "Thrissur",
            "Kannur"
        ],

        "format": [
            "Large",
            "Medium",
            "Medium",
            "Medium",
            "Small"
        ],

        "traffic_index": [
            1.30,
            1.05,
            0.90,
            1.00,
            0.78
        ]
    })

    stores.to_csv(
        DATA_DIR / "stores.csv",
        index=False
    )

    return stores


# ==========================================================
# CALENDAR
# ==========================================================

def generate_calendar():

    dates = pd.date_range(
        start="2025-01-01",
        end="2026-09-23",
        freq="D"
    )

    calendar = pd.DataFrame({
        "date": dates
    })

    calendar["year"] = calendar["date"].dt.year

    calendar["month"] = calendar["date"].dt.month

    calendar["day"] = calendar["date"].dt.day

    calendar["dayofweek"] = calendar["date"].dt.dayofweek

    calendar["weekend"] = (
        calendar["dayofweek"] >= 5
    ).astype(int)

    calendar["week_of_year"] = (
        calendar["date"].dt.isocalendar().week.astype(int)
    )

    # ------------------------------------------------------
    # Season
    # ------------------------------------------------------

    calendar["season"] = np.select(
        [
            calendar["month"].isin([1, 2]),
            calendar["month"].isin([3, 4, 5]),
            calendar["month"].isin([6, 7, 8, 9]),
            calendar["month"].isin([10, 11, 12])
        ],
        [
            "Winter",
            "Summer",
            "Monsoon",
            "Festive"
        ],
        default="Normal"
    )

    # ------------------------------------------------------
    # Festivals
    # ------------------------------------------------------

    calendar["festival"] = "None"

    festival_windows = [

        ("Vishu", "2025-04-12", "2025-04-16"),
        ("Onam", "2025-08-28", "2025-09-07"),
        ("Diwali", "2025-10-16", "2025-10-22"),
        ("Christmas", "2025-12-20", "2025-12-27"),

        ("Vishu", "2026-04-12", "2026-04-16"),
        ("Onam", "2026-08-20", "2026-08-30")
    ]

    for name, start, end in festival_windows:

        mask = (
            (calendar["date"] >= pd.Timestamp(start))
            &
            (calendar["date"] <= pd.Timestamp(end))
        )

        calendar.loc[
            mask,
            "festival"
        ] = name

    calendar["festival_flag"] = (
        calendar["festival"] != "None"
    ).astype(int)

    festival_lift = {
        "Vishu": 0.35,
        "Onam": 0.70,
        "Diwali": 0.55,
        "Christmas": 0.50,
        "None": 0.0
    }

    calendar["festival_lift"] = (
        calendar["festival"]
        .map(festival_lift)
        .fillna(0)
    )

    # ------------------------------------------------------
    # Weather
    # ------------------------------------------------------

    rainfall = []

    temperature = []

    for month in calendar["month"]:

        if month in [6, 7, 8, 9]:

            rain = rng.normal(
                75,
                25
            )

        else:

            rain = rng.normal(
                25,
                12
            )

        rainfall.append(
            max(0, rain)
        )

        if month in [3, 4, 5]:

            temp = rng.normal(
                32,
                2
            )

        else:

            temp = rng.normal(
                28,
                2
            )

        temperature.append(temp)

    calendar["rainfall_mm"] = np.round(
        rainfall,
        1
    )

    calendar["temperature_c"] = np.round(
        temperature,
        1
    )

    calendar["weather"] = np.select(
        [
            calendar["rainfall_mm"] > 90,
            calendar["rainfall_mm"] > 55,
            calendar["temperature_c"] > 32
        ],
        [
            "Heavy Rain",
            "Rain",
            "Hot"
        ],
        default="Normal"
    )

    calendar.to_csv(
        DATA_DIR / "calendar.csv",
        index=False
    )

    return calendar


# ==========================================================
# CUSTOMER PRODUCT PREFERENCE
# ==========================================================

def create_customer_preferences(
    customers,
    products
):

    preference = {}

    categories = products["category"].unique()

    for customer in customers.itertuples():

        preferred_category = rng.choice(
            categories,
            p=[
                0.55,
                0.20,
                0.15,
                0.10
            ]
        )

        available = products[
            products["category"]
            == preferred_category
        ]

        product = rng.choice(
            available["product_id"]
        )

        preference[
            customer.customer_id
        ] = product

    return preference


# ==========================================================
# SALES GENERATION
# ==========================================================

def generate_sales(
    calendar,
    customers,
    products,
    stores
):

    customer_preferences = (
        create_customer_preferences(
            customers,
            products
        )
    )

    preference_counts = pd.Series(
        list(customer_preferences.values())
    ).value_counts()

    rows = []

    for day in calendar.itertuples():

        for store in stores.itertuples():

            for product in products.itertuples():

                # ------------------------------------------
                # Base demand
                # ------------------------------------------

                demand = (
                    2.0
                    * product.demand_index
                    * store.traffic_index
                )

                # ------------------------------------------
                # Weekend
                # ------------------------------------------

                if day.weekend:

                    demand *= 1.18

                # ------------------------------------------
                # Season
                # ------------------------------------------

                seasonal_multiplier = {

                    "Winter": 1.05,

                    "Summer": 1.10,

                    "Monsoon": 0.90,

                    "Festive": 1.15
                }.get(
                    day.season,
                    1.0
                )

                demand *= seasonal_multiplier

                # ------------------------------------------
                # Festival
                # ------------------------------------------

                demand *= (
                    1
                    + day.festival_lift
                )

                # ------------------------------------------
                # Weather
                # ------------------------------------------

                if day.weather == "Heavy Rain":

                    demand *= 0.82

                elif day.weather == "Rain":

                    demand *= 0.93

                elif day.weather == "Hot":

                    demand *= 1.03

                # ------------------------------------------
                # Customer preference
                # ------------------------------------------

                popularity = (
                    preference_counts
                    .get(
                        product.product_id,
                        0
                    )
                )

                demand *= (
                    1
                    + popularity / 500
                )

                # ------------------------------------------
                # Competitor price
                # ------------------------------------------

                competitor_price = (
                    product.base_price
                    *
                    rng.uniform(
                        0.94,
                        1.07
                    )
                )

                # ------------------------------------------
                # Promotion
                # ------------------------------------------

                promotion_probability = (
                    0.08
                    + 0.18 * day.festival_flag
                    + 0.04 * day.weekend
                )

                promotion = int(
                    rng.random()
                    < promotion_probability
                )

                if promotion:

                    discount = rng.uniform(
                        0.05,
                        0.25
                    )

                else:

                    discount = 0.0

                selling_price = (
                    product.base_price
                    * (1 - discount)
                )

                # ------------------------------------------
                # Price elasticity
                # ------------------------------------------

                price_ratio = (
                    selling_price
                    /
                    competitor_price
                )

                if price_ratio < 0.97:

                    demand *= 1.12

                elif price_ratio > 1.03:

                    demand *= 0.88

                # ------------------------------------------
                # Random demand variation
                # ------------------------------------------

                demand *= rng.uniform(
                    0.85,
                    1.15
                )

                expected_demand = max(
                    0.1,
                    demand
                )

                demand_units = rng.poisson(
                    expected_demand
                )

                # ------------------------------------------
                # Inventory
                # ------------------------------------------

                inventory = int(
                    max(
                        0,
                        rng.normal(
                            18
                            +
                            10 * store.traffic_index,
                            7
                        )
                    )
                )

                units_sold = min(
                    demand_units,
                    inventory
                )

                stockout = int(
                    demand_units > inventory
                )

                revenue = (
                    units_sold
                    * selling_price
                )

                cost = (
                    revenue
                    * (1 - product.margin_rate)
                )

                profit = (
                    revenue
                    - cost
                )

                # ------------------------------------------
                # Customer
                # ------------------------------------------

                if units_sold > 0:

                    customer_id = rng.choice(
                        customers["customer_id"]
                    )

                else:

                    customer_id = None

                rows.append({

                    "date": day.date,

                    "store_id":
                        store.store_id,

                    "product_id":
                        product.product_id,

                    "customer_id":
                        customer_id,

                    "selling_price":
                        round(
                            selling_price,
                            2
                        ),

                    "competitor_price":
                        round(
                            competitor_price,
                            2
                        ),

                    "promotion":
                        promotion,

                    "discount_rate":
                        round(
                            discount,
                            3
                        ),

                    "demand_units":
                        demand_units,

                    "units_sold":
                        units_sold,

                    "inventory_available":
                        inventory,

                    "stockout":
                        stockout,

                    "festival":
                        day.festival,

                    "festival_flag":
                        day.festival_flag,

                    "season":
                        day.season,

                    "rainfall_mm":
                        day.rainfall_mm,

                    "temperature_c":
                        day.temperature_c,

                    "weather":
                        day.weather,

                    "revenue":
                        round(
                            revenue,
                            2
                        ),

                    "profit":
                        round(
                            profit,
                            2
                        )
                })

    sales = pd.DataFrame(rows)

    sales.to_csv(
        DATA_DIR / "sales_daily.csv",
        index=False
    )

    return sales


# ==========================================================
# MAIN
# ==========================================================

def main():

    print(
        "\n"
        "========================================\n"
        " RETAIL DEMAND DATA SIMULATOR\n"
        "========================================\n"
    )

    print("Generating customers...")

    customers = generate_customers()

    print(
        f"Customers: {len(customers):,}"
    )

    print("Generating products...")

    products = generate_products()

    print(
        f"Products: {len(products):,}"
    )

    print("Generating stores...")

    stores = generate_stores()

    print(
        f"Stores: {len(stores):,}"
    )

    print("Generating calendar...")

    calendar = generate_calendar()

    print(
        f"Calendar days: {len(calendar):,}"
    )

    print(
        "Generating sales and inventory data..."
    )

    sales = generate_sales(
        calendar,
        customers,
        products,
        stores
    )

    print(
        f"Sales rows: {len(sales):,}"
    )

    print(
        "\nData generation completed."
    )

    print(
        "\nFiles created:"
    )

    for file in DATA_DIR.glob("*.csv"):

        df = pd.read_csv(file)

        print(
            f"{file.name:<25} "
            f"{df.shape[0]:>10,} rows "
            f"{df.shape[1]:>3} columns"
        )


if __name__ == "__main__":
    main()