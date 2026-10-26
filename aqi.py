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


if __name__ == "__main__":
    for value in [5, 20, 40, 100, 180, 300]:
        print(value, "->", pm25_to_category(value))
