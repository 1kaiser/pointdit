// One-off test driver: uploads a real, different image via the actual #imageInput file control
// (not a code-path shortcut) and confirms the page reports a genuinely different result.
const { chromium } = require('playwright-core');
const path = require('path');

async function main() {
  const url = process.argv[2] || 'http://127.0.0.1:8080/index.html';
  const imagePath = process.argv[3];
  const outPng = process.argv[4] || '/tmp/hosted_upload_screenshot.png';

  const browser = await chromium.launch({
    executablePath: process.env.CHROME_PATH || '/home/kaiser/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome',
    headless: true,
    args: [
      '--no-sandbox', '--headless=new', '--use-angle=vulkan',
      '--enable-features=Vulkan', '--disable-vulkan-surface',
      '--enable-unsafe-webgpu', '--enable-webgpu-developer-features',
    ],
  });
  const page = await browser.newPage({ viewport: { width: 900, height: 900 } });
  page.on('console', (msg) => console.log('[console]', msg.text()));
  page.on('pageerror', (err) => console.log('[pageerror]', err.message));

  await page.goto(url, { waitUntil: 'load', timeout: 30000 });
  await page.waitForFunction(() => window.__demoDone === true, undefined, { timeout: 120000 });
  const defaultResult = await page.evaluate(() => window.__demoResult);
  console.log('DEFAULT_RESULT:', JSON.stringify(defaultResult));

  // Real file upload through the actual <input type="file"> element.
  await page.setInputFiles('#imageInput', path.resolve(imagePath));
  await page.waitForFunction(
    () => window.__demoResult && window.__demoResult.uploadedImage !== undefined,
    undefined,
    { timeout: 120000, polling: 500 },
  );
  const uploadResult = await page.evaluate(() => window.__demoResult);
  console.log('UPLOAD_RESULT:', JSON.stringify(uploadResult));

  await page.waitForTimeout(1500); // let model-viewer finish its own render after src swap
  await page.screenshot({ path: outPng, fullPage: true });
  console.log('screenshot saved:', outPng);
  await browser.close();
}

main().catch((e) => { console.error(e); process.exit(1); });
