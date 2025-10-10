from playwright.sync_api import sync_playwright, expect

def run(playwright):
    browser = playwright.chromium.launch(headless=True)
    context = browser.new_context()
    page = context.new_page()

    # Navigate to the app
    page.goto("http://localhost:8501")

    # Wait for the page to load
    page.wait_for_load_state("networkidle")

    # Click the link to the new page
    page.locator('a[href*="3_Diagnostic_Assistant"]').click(timeout=60000)

    # Wait for the page to load and check for the title
    expect(page).to_have_title("Diagnostic Assistant")
    expect(page.get_by_role("heading", name="💡 Diagnostic Assistant")).to_be_visible()

    # Take a screenshot
    page.screenshot(path="jules-scratch/verification/verification.png")

    browser.close()

with sync_playwright() as playwright:
    run(playwright)