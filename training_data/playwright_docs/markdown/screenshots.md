Source: https://playwright.dev/docs/screenshots

  * [](/)
  * Guides
  * Screenshots



On this page

# Screenshots

## Introduction​

Here is a quick way to capture a screenshot and save it into a file:
    
    
    await page.screenshot({ path: 'screenshot.png' });  
    

[Screenshots API](/docs/api/class-page#page-screenshot) accepts many parameters for image format, clip area, quality, etc. Make sure to check them out.

## Full page screenshots​

Full page screenshot is a screenshot of a full scrollable page, as if you had a very tall screen and the page could fit it entirely.
    
    
    await page.screenshot({ path: 'screenshot.png', fullPage: true });  
    

## Capture into buffer​

Rather than writing into a file, you can get a buffer with the image and post-process it or pass it to a third party pixel diff facility.
    
    
    const buffer = await page.screenshot();  
    console.log(buffer.toString('base64'));  
    

## Element screenshot​

Sometimes it is useful to take a screenshot of a single element.
    
    
    await page.locator('.header').screenshot({ path: 'screenshot.png' });  
    
