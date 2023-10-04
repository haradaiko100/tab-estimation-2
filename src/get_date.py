from datetime import datetime

now = datetime.now()
today_formated = now.strftime("%Y%m%d_%H%M%S")  # "%d/%m/%Y %H:%M:%S"
print("Today's date: ", today_formated)
