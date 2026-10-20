# Menu transcription status

The high-resolution `Hostel Mess Menu Poster.png` is the primary source. Its breakfast, lunch, snacks, dinner, North/South seasonal tables, and timing sections were inspected in enlarged crops. The earlier source pack's blank placeholders were not imported.

The menu CSVs now contain **256 weekly-rotation records** (Breakfast 89, Lunch 63, Evening Snacks 32, Dinner 72) and **39 seasonal records**. Main menu records: 147 verified and 109 marked `needs_verification`. Seasonal records: 33 verified and 6 marked `needs_verification`. Blank and `pending_transcription` placeholders are filtered from display and export.

Printed breakfast and dinner day labels, including `Sunday 1` and `Sunday 2`, are preserved. The lunch and snacks tables show no weekday headings in the poster, so their 7 and 8 visible columns respectively are retained in source order as `Column 1`… entries and marked `needs_verification`; they are not assigned to dates. Repeated Sunday cycles also remain unassigned to calendar week numbers.

| Section | Status | Needed source |
|---|---|---|
| Breakfast weekly rotation | Transcribed | “Moong Pongani” is marked for spelling verification; category cells are kept as category labels, not dishes |
| Lunch North and South rotation | Transcribed, weekday mapping pending | Weekday headers are not visible; dish spellings noted in CSV where unclear |
| Evening Snacks rotation | Transcribed, weekday mapping pending | Weekday headers are not visible |
| Dinner North and South rotation | Transcribed | Mrapla Pulusu, Vimgaya Sambhar (Onion), Pottu Pesa Poriyal, Anikani Vepudu, Sore Biryani, Madh Ki Dal, Mater Moti, Mushroom Matar / Methi Methi Matar, Aloo Bengan, Aloo Shimla, Nutri Chilly, Soanp Kheer, and Moong Dal Halwa / Ice Cream are marked for text/spelling verification |
| North Cuisine seasonal vegetables | Transcribed | Tindkand Matar, Lauki/Tori Lahsodar, Baingan Bhartua Gobhi Gajar Matar, Aloo Gobhi Adzaki, Nuni Keema Matar, and Saoa Saag + Makaitori wording need checking in the noted cells |
| South Cuisine seasonal vegetables | Transcribed | No remaining unclear food names |
| Meal timings | Readable | Retained in `src/menu.py`; distinguish working days from Sundays/holidays |

The weekly pattern remains separate from calendar dates. Menu category, day/cycle, source filename, and verification status are retained in the CSVs. No rows were added to the synthetic meal dataset, and menu data is not used as a model feature.
