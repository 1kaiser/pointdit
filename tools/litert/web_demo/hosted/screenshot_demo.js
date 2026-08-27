const { chromium } = require('playwright-core');

async function main() {
  const url = process.argv[2] || 'http://127.0.0.1:8080/index.html';
  const outPng = process.argv[3] || '/tmp/hosted_demo_screenshot.png';
  const browser = await chromium.launch({
    executablePath: process.env.CHROME_PATH || '/home/kaiser/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome',
    headless: true,
    args: [
      '--no-sandbox', '--headless=new', '--use-angle=vulkan',
      '--enable-features=Vulkan', '--disable-vulkan-surface',
      '--enable-unsafe-webgpu', '--enable-webgpu-developer-features',
    ],
  });
  const page = await browser.newPage({ viewport: { width: 900, height: 500 } });
  await page.goto(url, { waitUntil: 'load', timeout: 30000 });
  await page.waitForFunction(() => window.__demoDone === true, undefined, { timeout: 120000 });
  await page.screenshot({ path: outPng });
  console.log('screenshot saved:', outPng);
  const result = await page.evaluate(() => window.__demoResult || null);
  console.log('RESULT:', JSON.stringify(result));
  await browser.close();
}

main().catch((e) => { console.error(e); process.exit(1); });
