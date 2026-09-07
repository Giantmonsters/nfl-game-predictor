"""
Visits the Streamlit app with a real headless browser (not a plain HTTP
request) so it behaves like a genuine visit. If the app is showing the
"gone to sleep" page, this finds and clicks the "Yes, get this app back
up!" button to actually wake it — something a plain curl request cannot
do, since waking requires real browser interaction, not just an HTTP GET.
"""
from playwright.sync_api import sync_playwright

APP_URL = "https://nfl-game-predictor-9bacfsbwugzbnwcptra946.streamlit.app/"

def main():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(APP_URL, timeout=60000)
        page.wait_for_timeout(3000)  # let the page settle

        # Look for the wake-up button by its visible text and click it if present
        wake_button = page.get_by_text("Yes, get this app back up!", exact=False)
        if wake_button.count() > 0:
            print("App was asleep — clicking wake-up button.")
            wake_button.first.click()
            page.wait_for_timeout(15000)  # give it time to actually spin up
            print("Waited for app to wake up.")
        else:
            print("App was already awake — no action needed.")

        browser.close()

if __name__ == "__main__":
    main()