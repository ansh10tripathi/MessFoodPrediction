# Exploratory Data Analysis: Synthetic Mess Food Dataset

> **Synthetic development data only.** These summaries describe generated sample records, not measured university operations and not real-world predictive performance.

## Dataset overview

- Dimensions: **1,460 rows × 15 columns**
- Columns: `date`, `meal_period`, `menu_category`, `expected_students`, `actual_students`, `exam_day`, `holiday`, `special_event`, `temperature_c`, `rainfall_mm`, `prepared_kg`, `consumed_kg`, `edible_leftover_kg`, `discarded_waste_kg`, `dataset_type`
- Date range: **2025-01-01 to 2025-12-31** (365 distinct dates)

## Records and data quality

### Records per meal

| Meal | Records |
|---|---:|
| Breakfast | 365 |
| Dinner | 365 |
| Lunch | 365 |
| Snacks | 365 |

- Missing cells: **0**
- Exact duplicate records: **0**
- Validator status: **VALID**

## Numerical summary statistics

Values are shown in the source units; student counts are people, temperatures °C, rainfall mm, and food quantities kg.

| Column | Count | Mean | Std. dev. | Min | Median | Max |
|---|---:|---:|---:|---:|---:|---:|
| `expected_students` | 1,460 | 445.150 | 152.916 | 57.000 | 427.500 | 730.000 |
| `actual_students` | 1,460 | 442.626 | 153.633 | 53.000 | 441.000 | 764.000 |
| `temperature_c` | 1,460 | 26.738 | 5.332 | 17.100 | 27.300 | 35.800 |
| `rainfall_mm` | 1,460 | 3.632 | 6.748 | 0.000 | 0.000 | 23.900 |
| `prepared_kg` | 1,460 | 199.525 | 112.153 | 12.475 | 163.351 | 475.514 |
| `consumed_kg` | 1,460 | 176.658 | 99.063 | 10.812 | 145.390 | 413.134 |
| `edible_leftover_kg` | 1,460 | 14.145 | 9.864 | 0.742 | 11.771 | 54.579 |
| `discarded_waste_kg` | 1,460 | 8.722 | 6.254 | 0.366 | 7.181 | 39.601 |

## Food totals

Waste percentage is defined here as **total discarded_waste_kg ÷ total prepared_kg × 100**. Edible leftovers remain a separate category and are not included in discarded waste.

| Quantity | Total (kg) |
|---|---:|
| Prepared | 291,306.661 |
| Consumed | 257,920.780 |
| Edible leftovers | 20,651.103 |
| Discarded waste | 12,734.778 |
| Discarded waste / prepared | 4.37% |

## Average consumption and discarded waste by meal

| Meal | Records | Mean consumed (kg/record) | Mean discarded (kg/record) |
|---|---:|---:|---:|
| Breakfast | 365 | 123.197 | 6.221 |
| Dinner | 365 | 233.044 | 11.275 |
| Lunch | 365 | 289.397 | 14.300 |
| Snacks | 365 | 60.995 | 3.094 |

## Interpretation and limitations

The dataset is programmatically generated with a fixed seed. Its totals and averages describe only the generator's assumptions. They must not be presented as actual mess measurements or as evidence of model accuracy. Synthetic weather values are illustrative, not observed forecasts or records.
