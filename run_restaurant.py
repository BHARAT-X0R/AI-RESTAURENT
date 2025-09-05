# ===================================================================
# AI RESTAURANT - V9.5 - FINAL RELEASE
# ===================================================================
# This definitive version features a compact, horizontal order display
# on the main screen for a cleaner UI, and includes all features.
# ===================================================================

import cv2
import numpy as np
from datetime import datetime, timedelta
from deepface import DeepFace
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import mediapipe as mp

# ===================================================================
# 1. CAMERA SELECTION UTILITY
# ===================================================================
def select_camera():
    """Scans for available cameras and prompts the user to select one."""
    print("Scanning for available cameras...")
    available_cameras = []
    for i in range(10):
        cap_test = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap_test.isOpened():
            available_cameras.append(i)
            cap_test.release()
    if not available_cameras:
        print("FATAL ERROR: No cameras found on your system."); exit()
    print(f"Found available cameras: {available_cameras}")
    if len(available_cameras) == 1:
        print(f"Automatically selecting the only available camera: {available_cameras[0]}")
        return available_cameras[0]
    while True:
        try:
            choice = int(input(f"Please enter the camera number you wish to use: "))
            if choice in available_cameras:
                print(f"You selected camera {choice}."); return choice
            else:
                print(f"Invalid choice. Please select from {available_cameras}.")
        except ValueError:
            print("Invalid input. Please enter a number.")

# ===================================================================
# 2. CONFIGURATION
# ===================================================================
RECOGNITION_INTERVAL_SECONDS = 10; CUSTOMER_TIMEOUT_SECONDS = 10; TAX_RATE = 0.05; TOTAL_TABLES = 10; KITCHEN_CAPACITY = 5; DASHBOARD_REFRESH_INTERVAL_SECONDS = 30
DB_PATH = "face_database"; SESSION_FILE = "session.json"; LOG_FILE = "log.csv"; BILLS_DIR = "bills"
MAX_WAIT_SECONDS = 600
C_BLACK = (15, 15, 15); C_DARK_GRAY = (40, 40, 40); C_MED_GRAY = (90, 90, 90); C_LIGHT_GRAY = (210, 210, 210); C_BLUE = (255, 190, 100); C_YELLOW = (0, 220, 255); C_GREEN = (100, 255, 100); C_RED = (100, 100, 255); C_WHITE = (255, 255, 255)
STATUS_COLORS = {"Ordering": C_BLUE, "Waiting for Food": C_YELLOW, "Served": C_GREEN}
MENU_PRICES = { "Biryani": 200.00, "Burger": 60.00, "Chapathi": 20.00, "Chicken": 70.00, "Coke": 20.00, "Dosa": 30.00, "Fries": 60.00, "Idly": 30.00, "Panipuri": 30.00, "Pizza": 70.00, "Rice": 80.00, "Samosa": 10.00, "Vada": 20.00, "Water": 20.00 }
MENU_ITEMS = list(MENU_PRICES.keys()); ACTION_BUTTONS = ["Confirm Order", "Mark as Served", "Task Completed", "Close & Report"]

# ===================================================================
# 3. GLOBAL VARIABLES & STATE MANAGEMENT
# ===================================================================
customer_data = {}; table_status = {i: "Unoccupied" for i in range(1, TOTAL_TABLES + 1)}; unknown_customer_count = 0; selected_customer = None; avg_item_prep_times = {}; last_model_update = datetime.now() - timedelta(minutes=6); last_dashboard_update = datetime.now() - timedelta(seconds=DASHBOARD_REFRESH_INTERVAL_SECONDS)
should_quit = False
MENU_WINDOW_NAME = "Order Menu"; MENU_ITEM_HEIGHT = 35; MENU_PADDING = 15; MENU_WIDTH = 450
ACTION_GRID_ROWS = 2; TABLE_GRID_ROWS = 2; ACTION_SECTION_HEIGHT = ACTION_GRID_ROWS * 50 + MENU_PADDING; TABLE_SECTION_HEIGHT = TABLE_GRID_ROWS * 50 + MENU_PADDING * 2
MENU_HEIGHT = (len(MENU_ITEMS) * MENU_ITEM_HEIGHT) + ACTION_SECTION_HEIGHT + TABLE_SECTION_HEIGHT + (MENU_PADDING * 8)
menu_bg = np.full((int(MENU_HEIGHT), MENU_WIDTH, 3), C_BLACK, dtype="uint8")

# ===================================================================
# 4. HELPER FUNCTIONS
# ===================================================================
def log_event(event, customer="", details="", value=""):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S'); log_details = f'"{details}"'
    with open(LOG_FILE, 'a', newline='') as f: f.write(f'"{timestamp}","{event}","{customer}",{log_details},"{value}"\n')

def update_wait_time_model():
    global avg_item_prep_times, last_model_update; print("Updating predictive model...")
    try:
        if not os.path.exists(LOG_FILE) or os.path.getsize(LOG_FILE) < 10: return
        df = pd.read_csv(LOG_FILE); df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        confirmed = df[df['Event'] == 'ORDER_CONFIRMED'].set_index('Customer'); served = df[df['Event'] == 'ORDER_SERVED'].set_index('Customer')
        merged = confirmed.join(served, lsuffix='_c', rsuffix='_s'); merged.dropna(subset=['Timestamp_s', 'Details_c'], inplace=True)
        merged['PrepTime'] = (merged['Timestamp_s'] - merged['Timestamp_c']).dt.total_seconds()
        item_times = {}
        for _, row in merged.iterrows():
            for item in str(row['Details_c']).split(';'):
                if item:
                    if item not in item_times: item_times[item] = []
                    item_times[item].append(row['PrepTime'])
        for item, times in item_times.items():
            if times: avg_item_prep_times[item] = sum(times) / len(times)
        last_model_update = datetime.now()
    except Exception as e: print(f"Model update failed: {e}")

def calculate_order_eta(order_items):
    if not order_items: return 0
    return max(avg_item_prep_times.get(item, 90) for item in order_items)
    
def detect_faces_mediapipe(image, face_detector):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB); results = face_detector.process(img_rgb)
    extracted_faces = []
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box; ih, iw, _ = image.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            x, y = max(0, x), max(0, y)
            extracted_faces.append({'facial_area': {'x': x, 'y': y, 'w': w, 'h': h}, 'confidence': detection.score[0]})
    return extracted_faces

def generate_dashboard_figure():
    if not os.path.exists(LOG_FILE): return None
    try:
        df = pd.read_csv(LOG_FILE);
        if df.empty: return None
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    except Exception: return None
    fig, axes = plt.subplots(2, 2, figsize=(18, 10)); fig.suptitle('AI Restaurant Analytics Dashboard', fontsize=20); sns.set_style("whitegrid")
    peak_hours = df[df['Event'] == 'NEW_CUSTOMER']['Timestamp'].dt.hour
    sns.countplot(ax=axes[0, 0], x=peak_hours, palette="viridis", order=range(0, 24)); axes[0, 0].set_title('Peak Hours'); axes[0, 0].set_xlabel('Hour'); axes[0, 0].set_ylabel('Customers')
    orders_df = df[df['Event'] == 'ORDER_CONFIRMED'].copy()
    if not orders_df.empty:
        all_items = [item for sublist in orders_df['Details'].dropna().str.split(';') for item in sublist if item]
        if all_items:
            top_items = Counter(all_items).most_common(10)
            if top_items:
                item_df = pd.DataFrame(top_items, columns=['Dish', 'Count']); sns.barplot(ax=axes[0, 1], x='Count', y='Dish', data=item_df, palette="plasma")
    axes[0, 1].set_title('Most Popular Dishes')
    checkout_df = df[df['Event'] == 'CHECKOUT'].copy()
    if not checkout_df.empty:
        checkout_df['Details'] = pd.to_numeric(checkout_df['Details'], errors='coerce'); checkout_df['Date'] = checkout_df['Timestamp'].dt.date
        daily_revenue = checkout_df.groupby('Date')['Details'].sum(); daily_revenue.plot(ax=axes[1, 0], kind='line', marker='o', linestyle='-')
    axes[1, 0].set_title('Daily Revenue Trend'); axes[1, 0].set_ylabel('Revenue (Rs.)'); axes[1, 0].tick_params(axis='x', rotation=30)
    avg_wait_seconds = pd.to_numeric(checkout_df['Value'], errors='coerce').mean(); total_revenue = pd.to_numeric(checkout_df['Details'], errors='coerce').sum()
    axes[1, 1].axis('off'); kpi_text = f"--- Key Performance Indicators ---\n\nTotal Revenue Recorded: Rs. {total_revenue:.2f}\n\n"
    if not pd.isna(avg_wait_seconds): kpi_text += f"Average Customer Visit Time:\n{int(avg_wait_seconds // 60)} minutes, {int(avg_wait_seconds % 60)} seconds"
    axes[1, 1].text(0.5, 0.5, kpi_text, ha='center', va='center', fontsize=14, wrap=True); axes[1, 1].set_title('Key Metrics')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); return fig

def save_session():
    session_data = {'customers': {}, 'tables': table_status}
    for name, data in customer_data.items():
        session_data['customers'][name] = data.copy()
        for key, value in data.items():
            if isinstance(value, datetime): session_data['customers'][name][key] = value.isoformat()
            if isinstance(value, timedelta): session_data['customers'][name][key] = value.total_seconds()
    with open(SESSION_FILE, 'w') as f: json.dump(session_data, f, indent=4)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Session saved to {SESSION_FILE}")

def load_session():
    global customer_data, unknown_customer_count, table_status
    if os.path.exists(SESSION_FILE):
        with open(SESSION_FILE, 'r') as f:
            session_data = json.load(f); table_status = {int(k): v for k, v in session_data.get('tables', {}).items()}
            for name, data in session_data.get('customers', {}).items():
                customer_data[name] = data
                for key, value in data.items():
                    if key in ['entry_time', 'last_seen', 'recognition_timeout']: customer_data[name][key] = datetime.fromisoformat(value)
                    if key == 'eta': customer_data[name][key] = timedelta(seconds=value)
                if name.startswith("Customer-"):
                    num = int(name.split('-')[1]);
                    if num > unknown_customer_count: unknown_customer_count = num
            print(f"Session loaded successfully from {SESSION_FILE}")

def generate_bill_file(customer_name, data):
    if not data.get('table_number'): print(f"Cannot generate bill for {customer_name}: No table number assigned."); return
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = os.path.join(BILLS_DIR, f"Table_{data['table_number']}_Bill_{timestamp}.txt")
    bill_content = ["="*30, "      AI RESTAURANT RECEIPT", "="*30, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", f"Table #: {data['table_number']}", f"Customer: {customer_name}", "-"*30, "ITEMS:"]
    subtotal = sum(MENU_PRICES.get(item, 0) for item in data.get('orders', {}).keys())
    for item in data.get('orders', {}).keys(): bill_content.append(f"  - {item:<15} Rs.{MENU_PRICES.get(item, 0):>7.2f}")
    tax = subtotal * TAX_RATE; grand_total = subtotal + tax
    bill_content.extend(["-"*30, f"{'Subtotal:':<17} Rs.{subtotal:>7.2f}", f"{'Tax (5%):':<17} Rs.{tax:>7.2f}", "="*30, f"{'GRAND TOTAL:':<17} Rs.{grand_total:>7.2f}", "\nThank you for dining with us!"])
    with open(filename, 'w') as f: f.write("\n".join(bill_content))
    print(f"Bill for {customer_name} at Table {data['table_number']} saved to {filename}")

def calculate_daily_total():
    print("\n" + "="*40); print("      GENERATING DAILY SALES REPORT"); print("="*40)
    try:
        if not os.path.exists(LOG_FILE): print("No log file found. No sales recorded today."); return
        df = pd.read_csv(LOG_FILE); df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        today = datetime.now().date()
        todays_checkouts = df[(df['Event'] == 'CHECKOUT') & (df['Timestamp'].dt.date == today)]
        if todays_checkouts.empty: total_revenue = 0
        else: total_revenue = pd.to_numeric(todays_checkouts['Details'], errors='coerce').sum()
        print(f"Date: {today.strftime('%Y-%m-%d')}"); print(f"Total Customers Checked Out: {len(todays_checkouts)}"); print(f"TOTAL REVENUE FOR THE DAY: Rs. {total_revenue:.2f}"); print("="*40 + "\n")
    except Exception as e: print(f"Could not generate report: {e}")

# ===================================================================
# 5. MOUSE CALLBACKS & GUI DRAWING
# ===================================================================
def select_customer_callback(event, x, y, flags, param):
    global selected_customer
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_on_customer = False
        for name, data in customer_data.items():
            if 'last_box' in data:
                (fx, fy, fw, fh) = data['last_box']
                if fx < x < fx + fw and fy < y < fy + fh: selected_customer = name; clicked_on_customer = True; break
        if not clicked_on_customer: selected_customer = None

def menu_click_callback(event, x, y, flags, param):
    global customer_data, selected_customer, table_status, should_quit
    if not (event == cv2.EVENT_LBUTTONDOWN): return
    action_section_start_y = (len(MENU_ITEMS) * MENU_ITEM_HEIGHT) + (MENU_PADDING * 5)
    for i, action in enumerate(ACTION_BUTTONS):
        row = i // 2; col = i % 2; button_x = MENU_PADDING + (col * (MENU_WIDTH // 2)); button_y = action_section_start_y + (row * 50) + MENU_PADDING; button_w = MENU_WIDTH // 2 - MENU_PADDING - 5
        if button_x < x < button_x + button_w and button_y < y < button_y + 40:
            if action == "Close & Report": should_quit = True; return
            if selected_customer and selected_customer in customer_data:
                current_customer = customer_data[selected_customer]
                if action == "Confirm Order" and current_customer['status'] == "Ordering":
                    current_customer['status'] = "Waiting for Food"; order_details = ';'.join(current_customer['orders'].keys()); log_event("ORDER_CONFIRMED", selected_customer, order_details)
                    current_customer['eta'] = timedelta(seconds=calculate_order_eta(current_customer['orders'].keys()))
                elif action == "Mark as Served" and current_customer['status'] == "Waiting for Food":
                    current_customer['status'] = "Served"; log_event("ORDER_SERVED", selected_customer)
                elif action == "Task Completed":
                    generate_bill_file(selected_customer, current_customer); wait_time_total = (datetime.now() - current_customer['entry_time']).total_seconds()
                    log_event("CHECKOUT", selected_customer, current_customer.get('bill',0), int(wait_time_total))
                    table_num = current_customer.get('table_number');
                    if table_num and table_num in table_status: table_status[table_num] = "Unoccupied"
                    del customer_data[selected_customer]; selected_customer = None
                return
    if not (selected_customer and selected_customer in customer_data): return
    current_customer = customer_data[selected_customer]
    for i, item in enumerate(MENU_ITEMS):
        item_y = (i * MENU_ITEM_HEIGHT) + MENU_PADDING * 4
        if MENU_PADDING < x < MENU_WIDTH - MENU_PADDING and item_y < y < item_y + MENU_ITEM_HEIGHT:
            if current_customer['status'] != "Served":
                if current_customer['status'] == "Waiting for Food": current_customer['status'] = "Ordering"; log_event("ORDER_MODIFIED", selected_customer)
                if item in current_customer['orders']: del current_customer['orders'][item]
                else: current_customer['orders'][item] = 1
            return
    table_section_start_y = ((len(MENU_ITEMS) * MENU_ITEM_HEIGHT) + (MENU_PADDING * 5)) + ACTION_SECTION_HEIGHT + MENU_PADDING
    for i in range(1, TOTAL_TABLES + 1):
        row = (i - 1) // 5; col = (i - 1) % 5
        table_x = MENU_PADDING + (col * 85); table_y = table_section_start_y + 20 + (row * 50)
        if table_x < x < table_x + 75 and table_y < y < table_y + 40:
            if table_status[i] == "Unoccupied":
                old_table = current_customer.get('table_number');
                if old_table: table_status[old_table] = "Unoccupied"
                current_customer['table_number'] = i; table_status[i] = selected_customer
                log_event("TABLE_ASSIGNED", selected_customer, details=str(i))
            elif table_status[i] == selected_customer:
                current_customer['table_number'] = None; table_status[i] = "Unoccupied"
                log_event("TABLE_UNASSIGNED", selected_customer, details=str(i))
            return

def draw_menu():
    menu_window = menu_bg.copy(); title = f"Ordering for: {selected_customer}" if selected_customer else "No Customer Selected"; status_text = f"Status: {customer_data[selected_customer]['status']}" if selected_customer and selected_customer in customer_data else ""
    cv2.putText(menu_window, title, (MENU_PADDING, MENU_PADDING * 2), cv2.FONT_HERSHEY_DUPLEX, 0.7, C_WHITE, 1); cv2.putText(menu_window, status_text, (MENU_PADDING, MENU_PADDING * 3 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_LIGHT_GRAY, 1)
    for i, item in enumerate(MENU_ITEMS):
        y_pos = (i * MENU_ITEM_HEIGHT) + MENU_PADDING * 4; is_ordered = selected_customer and selected_customer in customer_data and item in customer_data[selected_customer]['orders']
        box_x1, box_y1 = MENU_PADDING, y_pos + 5; box_x2, box_y2 = MENU_PADDING + 20, y_pos + 25
        cv2.rectangle(menu_window, (box_x1, box_y1), (box_x2, box_y2), C_MED_GRAY, 1)
        if is_ordered: pt1 = (box_x1 + 4, box_y1 + 10); pt2 = (box_x1 + 8, box_y1 + 15); pt3 = (box_x1 + 16, box_y1 + 5); cv2.line(menu_window, pt1, pt2, C_GREEN, 2); cv2.line(menu_window, pt2, pt3, C_GREEN, 2)
        cv2.putText(menu_window, f"{item}", (MENU_PADDING + 30, y_pos + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_LIGHT_GRAY, 1)
    separator1_y = (len(MENU_ITEMS) * MENU_ITEM_HEIGHT) + (MENU_PADDING * 5); cv2.line(menu_window, (MENU_PADDING, separator1_y), (MENU_WIDTH - MENU_PADDING, separator1_y), (C_MED_GRAY), 1)
    action_section_start_y = separator1_y + MENU_PADDING
    for i, action in enumerate(ACTION_BUTTONS):
        row = i // 2; col = i % 2; button_x = MENU_PADDING + (col * (MENU_WIDTH // 2)); button_y = action_section_start_y + (row * 50); button_w = MENU_WIDTH // 2 - MENU_PADDING - 5
        button_color = C_RED if action == "Close & Report" else C_DARK_GRAY
        cv2.rectangle(menu_window, (button_x, button_y), (button_x + button_w, button_y + 40), button_color, -1)
        cv2.rectangle(menu_window, (button_x, button_y), (button_x + button_w, button_y + 40), C_MED_GRAY, 1)
        (tw, th), _ = cv2.getTextSize(action, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1); cv2.putText(menu_window, action, (button_x + (button_w - tw) // 2, button_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_LIGHT_GRAY, 1)
    separator2_y = action_section_start_y + ACTION_SECTION_HEIGHT
    cv2.line(menu_window, (MENU_PADDING, separator2_y), (MENU_WIDTH - MENU_PADDING, separator2_y), (C_MED_GRAY), 1)
    table_section_start_y = separator2_y + MENU_PADDING
    cv2.putText(menu_window, "Assign Table:", (MENU_PADDING, table_section_start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_WHITE, 1)
    for i in range(1, TOTAL_TABLES + 1):
        row = (i - 1) // 5; col = (i - 1) % 5; table_x = MENU_PADDING + (col * 85); table_y = table_section_start_y + 20 + (row * 50); status = table_status.get(i, "Error"); color = C_GREEN if status == "Unoccupied" else C_RED
        cv2.rectangle(menu_window, (table_x, table_y), (table_x + 75, table_y + 40), color, -1)
        cv2.rectangle(menu_window, (table_x, table_y), (table_x + 75, table_y + 40), C_MED_GRAY, 1)
        table_text = f"T{i}" if status == "Unoccupied" else f"T{i}:{str(status)[:4]}"
        (tw, th), _ = cv2.getTextSize(table_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.putText(menu_window, table_text, (table_x + (75-tw)//2, table_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_WHITE, 1)
    cv2.imshow(MENU_WINDOW_NAME, menu_window)

# ===================================================================
# 6. INITIALIZATION & MAIN LOOP
# ===================================================================
CAMERA_SOURCE = select_camera()
LIVE_VIEW_WINDOW_NAME = "AI Restaurant - Live View"
cv2.namedWindow(LIVE_VIEW_WINDOW_NAME); cv2.namedWindow(MENU_WINDOW_NAME)
cv2.setMouseCallback(LIVE_VIEW_WINDOW_NAME, select_customer_callback); cv2.setMouseCallback(MENU_WINDOW_NAME, menu_click_callback)
load_session()
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'w', newline='') as f: f.write("Timestamp,Event,Customer,Details,Value\n")
if not os.path.exists(BILLS_DIR): os.makedirs(BILLS_DIR)
cap = cv2.VideoCapture(CAMERA_SOURCE, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("❌ DSHOW failed. Trying default backend..."); cap = cv2.VideoCapture(CAMERA_SOURCE)
    if not cap.isOpened(): print("❌ FATAL ERROR: Cannot open selected camera. Exiting."); exit()
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
print(f"\n--- Starting Live System V9.5 (Final Polish) ---")

while True:
    if should_quit: break
    ret, frame = cap.read()
    if not ret: break
    current_time = datetime.now()
    if current_time - last_model_update > timedelta(minutes=5): update_wait_time_model()
    detected_faces = detect_faces_mediapipe(frame, face_detector)
    customers_in_frame = set(); unmatched_faces = list(range(len(detected_faces)))
    for name, data in customer_data.items():
        (lx, ly, lw, lh) = data.get('last_box', (0,0,0,0)); best_match_index = -1
        for i in unmatched_faces:
            face_box = detected_faces[i]['facial_area']; fx, fy, fw, fh = face_box['x'], face_box['y'], face_box['w'], face_box['h']
            if lx < fx + fw and lx + lw > fx and ly < fy + fh and ly + lh > fy: best_match_index = i; break
        if best_match_index != -1:
            customers_in_frame.add(name); unmatched_faces.remove(best_match_index)
            face_box = detected_faces[best_match_index]['facial_area']; data['last_box'] = (face_box['x'], face_box['y'], face_box['w'], face_box['h'])
            data['last_seen'] = current_time
            if current_time > data.get('recognition_timeout', current_time): data['recognition_timeout'] = current_time + timedelta(seconds=RECOGNITION_INTERVAL_SECONDS)
    for i in unmatched_faces:
        face_obj = detected_faces[i];
        if face_obj['confidence'] < 0.5: continue
        face_box = face_obj['facial_area']; fx, fy, fw, fh = face_box['x'], face_box['y'], face_box['w'], face_box['h']
        customer_name = None
        try:
            dfs = DeepFace.find(img_path=frame[fy:fy+fh, fx:fx+fw], db_path=DB_PATH, enforce_detection=False, silent=True)
            if dfs and not dfs[0].empty: customer_name = os.path.basename(dfs[0]['identity'][0]).split('.')[0].capitalize()
        except: pass
        if not customer_name: unknown_customer_count += 1; customer_name = f"Customer-{unknown_customer_count}"
        customers_in_frame.add(customer_name)
        customer_data[customer_name] = {"entry_time": current_time, "orders": {}, "bill": 0.0, "status": "Ordering", "last_box": (fx, fy, fw, fh), "last_seen": current_time, "recognition_timeout": current_time + timedelta(seconds=RECOGNITION_INTERVAL_SECONDS), "table_number": None}
        log_event("NEW_CUSTOMER", customer_name)
    customers_to_remove = []
    for name, data in customer_data.items():
        if name not in customers_in_frame and current_time - data.get('last_seen', current_time) > timedelta(seconds=CUSTOMER_TIMEOUT_SECONDS): customers_to_remove.append(name)
    for name in customers_to_remove:
        if name in customer_data: del customer_data[name]
    active_orders_count = 0; draw_menu()
    for name, data in customer_data.items():
        if not data.get('last_box'): continue
        if data.get('status') == 'Waiting for Food': active_orders_count += 1
        wait_time = current_time - data["entry_time"]
        if name == selected_customer: box_color = C_WHITE
        elif wait_time.total_seconds() > MAX_WAIT_SECONDS: box_color = C_RED
        else: box_color = STATUS_COLORS.get(data['status'], C_LIGHT_GRAY)
        wait_time_color = C_RED if wait_time.total_seconds() > MAX_WAIT_SECONDS else C_YELLOW
        data["bill"] = sum(MENU_PRICES.get(item, 0) for item in data["orders"].keys())
        (x, y, w, h) = data['last_box']; cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)
        
        # --- MODIFIED: Horizontal Order Display Logic ---
        info_y = y + h
        panel_height = 90 # Fixed height for 4 lines
        cv2.rectangle(frame, (x, info_y), (x + w, info_y + panel_height), C_DARK_GRAY, -1)

        table_text = f" | Tbl: {data.get('table_number')}" if data.get('table_number') else ""
        line1 = f"{name} [{data['status']}]{table_text}"
        
        eta_text = "";
        if 'eta' in data and data['status'] == 'Waiting for Food':
            eta_rem = data['eta'] - (current_time - data['entry_time']);
            if eta_rem.total_seconds() > 0: eta_text = f" | ETA: {int(eta_rem.total_seconds() // 60)}m"
        line2 = f"Wait: {str(timedelta(seconds=int(wait_time.total_seconds())))}{eta_text}"
        
        order_str = ", ".join(data["orders"].keys()) or "No order yet"
        line3 = f"Order: {order_str[:30]}" # Truncate if too long
        
        line4 = f"Bill: Rs. {data['bill']:.2f}"

        cv2.putText(frame, line1, (x + 5, info_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_WHITE, 1)
        cv2.putText(frame, line2, (x + 5, info_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, wait_time_color, 1)
        cv2.putText(frame, line3, (x + 5, info_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_LIGHT_GRAY, 1)
        cv2.putText(frame, line4, (x + 5, info_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_GREEN, 1)

    if active_orders_count >= KITCHEN_CAPACITY: 
        (tw, th), _ = cv2.getTextSize("KITCHEN OVERLOAD!", cv2.FONT_HERSHEY_DUPLEX, 1, 2)
        cv2.rectangle(frame, (frame.shape[1] - tw - 30, 10), (frame.shape[1] - 10, 50+th), C_DARK_GRAY, -1)
        cv2.putText(frame, "KITCHEN OVERLOAD!", (frame.shape[1] - tw - 20, 40+th//2), cv2.FONT_HERSHEY_DUPLEX, 1, C_RED, 2)
    cv2.imshow(LIVE_VIEW_WINDOW_NAME, frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == ord('s'): save_session()
    elif key == ord('d'):
        if current_time - last_dashboard_update > timedelta(seconds=DASHBOARD_REFRESH_INTERVAL_SECONDS):
            print("Generating dashboard..."); fig = generate_dashboard_figure()
            if fig: plt.show(); last_dashboard_update = datetime.now()
        else: print(f"Please wait {int(DASHBOARD_REFRESH_INTERVAL_SECONDS - (current_time - last_dashboard_update).total_seconds())}s before refreshing.")

# ===================================================================
# 7. CLEANUP
# ===================================================================
calculate_daily_total()
print("Shutting down... Saving final session state.")
save_session()
cap.release()
cv2.destroyAllWindows()
