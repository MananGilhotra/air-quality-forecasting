# aqi.py
# Map a PM2.5 value (µg/m³) to an AQI category.
# Standard used: US EPA PM2.5 AQI breakpoints (2024 update).
# Note: EPA defines these on 24-hour averages; we apply them to each hourly
# forecast as a simple category label.

AQI_CATEGORIES = [
    # (upper limit in µg/m³, category name, colour for plots)
    (9.0,          "Good",                           "green"),
    (35.4,         "Moderate",                       "yellow"),
    (55.4,         "Unhealthy for Sensitive Groups", "orange"),
    (125.4,        "Unhealthy",                      "red"),
    (225.4,        "Very Unhealthy",                 "purple"),
    (float("inf"), "Hazardous",                      "maroon"),
]


def pm25_to_category(value):
    for upper, name, colour in AQI_CATEGORIES:
        if value <= upper:
            return name


def category_accuracy(actual_values, predicted_values):
    """Share of hours where the forecast lands in the same AQI category as the truth."""
    actual_cats = [pm25_to_category(v) for v in actual_values]
    predicted_cats = [pm25_to_category(v) for v in predicted_values]
    matches = sum(a == p for a, p in zip(actual_cats, predicted_cats))
    return round(matches / len(actual_cats), 3)


def category_number(value):
    """Position of the category: 0 = Good, 1 = Moderate, ... 5 = Hazardous."""
    for number, (upper, name, colour) in enumerate(AQI_CATEGORIES):
        if value <= upper:
            return number


def within_one_category_accuracy(actual_values, predicted_values):
    """Share of hours where the forecast is in the correct AQI category
    or in the category right next to it (one step too high or too low)."""
    close = 0
    for actual, predicted in zip(actual_values, predicted_values):
        if abs(category_number(actual) - category_number(predicted)) <= 1:
            close += 1
    return round(close / len(actual_values), 3)


if __name__ == "__main__":
    for value in [5, 20, 40, 100, 180, 300]:
        print(value, "->", pm25_to_category(value))

    # Example: true value 40 (Unhealthy for Sensitive Groups)
    actual = [40, 40, 40]
    predicted = [45, 30, 130]   # same category, one category off, three categories off
    print("Exact category accuracy:", category_accuracy(actual, predicted))
    print("Within one category:", within_one_category_accuracy(actual, predicted))
