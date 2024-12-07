# Дані у вигляді частотних таблиць
data = {
    "Outlook": {"Sunny": {"Yes": 3, "No": 2}, "Rain": {"Yes": 2, "No": 2}, "Overcast": {"Yes": 4, "No": 0}},
    "Humidity": {"High": {"Yes": 3, "No": 4}, "Normal": {"Yes": 6, "No": 2}},
    "Wind": {"Weak": {"Yes": 6, "No": 2}, "Strong": {"Yes": 3, "No": 4}},
    "Play": {"Yes": 9, "No": 5}  # Загальна кількість випадків для Yes і No
}

# Ймовірності класів
total = sum(data["Play"].values())
P_Yes = data["Play"]["Yes"] / total
P_No = data["Play"]["No"] / total

# Ймовірності для заданих значень
P_Rain_Yes = data["Outlook"]["Rain"]["Yes"] / data["Play"]["Yes"]
P_Rain_No = data["Outlook"]["Rain"]["No"] / data["Play"]["No"]

P_High_Yes = data["Humidity"]["High"]["Yes"] / data["Play"]["Yes"]
P_High_No = data["Humidity"]["High"]["No"] / data["Play"]["No"]

P_Strong_Yes = data["Wind"]["Strong"]["Yes"] / data["Play"]["Yes"]
P_Strong_No = data["Wind"]["Strong"]["No"] / data["Play"]["No"]

# Розрахунок апостеріорних ймовірностей
P_Yes_given_data = P_Rain_Yes * P_High_Yes * P_Strong_Yes * P_Yes
P_No_given_data = P_Rain_No * P_High_No * P_Strong_No * P_No

# Нормалізація
P_Yes_normalized = P_Yes_given_data / (P_Yes_given_data + P_No_given_data)
P_No_normalized = P_No_given_data / (P_Yes_given_data + P_No_given_data)

# Результат
if P_Yes_normalized > P_No_normalized:
    result = "Yes, матч відбудеться"
else:
    result = "No, матч не відбудеться"

# Вивід результатів
print(f"Ймовірність 'Yes': {P_Yes_normalized:.2f}")
print(f"Ймовірність 'No': {P_No_normalized:.2f}")
print(f"Результат: {result}")
