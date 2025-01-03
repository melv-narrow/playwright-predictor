Source: https://playwright.dev/docs/aria-snapshots

  * [](/)
  * Guides
  * Aria snapshots



On this page

# Aria snapshots

## Overview​

In Playwright, aria snapshots provide a YAML representation of the accessibility tree of a page. These snapshots can be stored and compared later to verify if the page structure remains consistent or meets defined expectations.

The YAML format describes the hierarchical structure of accessible elements on the page, detailing **roles** , **attributes** , **values** , and **text content**. The structure follows a tree-like syntax, where each node represents an accessible element, and indentation indicates nested elements.

Following is a simple example of an aria snapshot for the playwright.dev homepage:
    
    
    - banner:  
      - heading /Playwright enables reliable/ [level=1]  
      - link "Get started"  
      - link "Star microsoft/playwright on GitHub"  
    - main:  
      - img "Browsers (Chromium, Firefox, WebKit)"  
      - heading "Any browser • Any platform • One API"  
    

Each accessible element in the tree is represented as a YAML node:
    
    
    - role "name" [attribute=value]  
    

  * **role** : Specifies the ARIA or HTML role of the element (e.g., `heading`, `list`, `listitem`, `button`).
  * **"name"** : Accessible name of the element. Quoted strings indicate exact values, `/patterns/` are used for regular expression.
  * **[attribute=value]** : Attributes and values, in square brackets, represent specific ARIA attributes, such as `checked`, `disabled`, `expanded`, `level`, `pressed`, or `selected`.



These values are derived from ARIA attributes or calculated based on HTML semantics. To inspect the accessibility tree structure of a page, use the [Chrome DevTools Accessibility Pane](https://developer.chrome.com/docs/devtools/accessibility/reference#pane).

## Snapshot matching​

The [expect(locator).toMatchAriaSnapshot()](/docs/api/class-locatorassertions#locator-assertions-to-match-aria-snapshot) assertion method in Playwright compares the accessible structure of the locator scope with a predefined aria snapshot template, helping validate the page's state against testing requirements.

For the following DOM:
    
    
    <h1>title</h1>  
    

You can match it using the following snapshot template:
    
    
    await expect(page.locator('body')).toMatchAriaSnapshot(`  
      - heading "title"  
    `);  
    

When matching, the snapshot template is compared to the current accessibility tree of the page:

  * If the tree structure matches the template, the test passes; otherwise, it fails, indicating a mismatch between expected and actual accessibility states.
  * The comparison is case-sensitive and collapses whitespace, so indentation and line breaks are ignored.
  * The comparison is order-sensitive, meaning the order of elements in the snapshot template must match the order in the page's accessibility tree.



### Partial matching​

You can perform partial matches on nodes by omitting attributes or accessible names, enabling verification of specific parts of the accessibility tree without requiring exact matches. This flexibility is helpful for dynamic or irrelevant attributes.
    
    
    <button>Submit</button>  
    

_aria snapshot_
    
    
    - button  
    

In this example, the button role is matched, but the accessible name ("Submit") is not specified, allowing the test to pass regardless of the button’s label.

* * *

For elements with ARIA attributes like `checked` or `disabled`, omitting these attributes allows partial matching, focusing solely on role and hierarchy.
    
    
    <input type="checkbox" checked>  
    

_aria snapshot for partial match_
    
    
    - checkbox  
    

In this partial match, the `checked` attribute is ignored, so the test will pass regardless of the checkbox state.

* * *

Similarly, you can partially match children in lists or groups by omitting specific list items or nested elements.
    
    
    <ul>  
      <li>Feature A</li>  
      <li>Feature B</li>  
      <li>Feature C</li>  
    </ul>  
    

_aria snapshot for partial match_
    
    
    - list  
      - listitem: Feature B  
    

Partial matches let you create flexible snapshot tests that verify essential page structure without enforcing specific content or attributes.

### Matching with regular expressions​

Regular expressions allow flexible matching for elements with dynamic or variable text. Accessible names and text can support regex patterns.
    
    
    <h1>Issues 12</h1>  
    

_aria snapshot with regular expression_
    
    
    - heading /Issues \d+/  
    

## Generating snapshots​

Creating aria snapshots in Playwright helps ensure and maintain your application’s structure. You can generate snapshots in various ways depending on your testing setup and workflow.

### 1\. Generating snapshots with the Playwright code generator​

If you’re using Playwright’s [Code Generator](/docs/codegen), generating aria snapshots is streamlined with its interactive interface:

  * **"Assert snapshot" Action** : In the code generator, you can use the "Assert snapshot" action to automatically create a snapshot assertion for the selected elements. This is a quick way to capture the aria snapshot as part of your recorded test flow.
  * **"Aria snapshot" Tab** : The "Aria snapshot" tab within the code generator interface visually represents the aria snapshot for a selected locator, letting you explore, inspect, and verify element roles, attributes, and accessible names to aid snapshot creation and review.



### 2\. Updating snapshots with `@playwright/test` and the `--update-snapshots` flag​

When using the Playwright test runner (`@playwright/test`), you can automatically update snapshots by running tests with the `--update-snapshots` flag:
    
    
    npx playwright test --update-snapshots  
    

This command regenerates snapshots for assertions, including aria snapshots, replacing outdated ones. It’s useful when application structure changes require new snapshots as a baseline. Note that Playwright will wait for the maximum expect timeout specified in the test runner configuration to ensure the page is settled before taking the snapshot. It might be necessary to adjust the `--timeout` if the test hits the timeout while generating snapshots.

#### Empty template for snapshot generation​

Passing an empty string as the template in an assertion generates a snapshot on-the-fly:
    
    
    await expect(locator).toMatchAriaSnapshot('');  
    

Note that Playwright will wait for the maximum expect timeout specified in the test runner configuration to ensure the page is settled before taking the snapshot. It might be necessary to adjust the `--timeout` if the test hits the timeout while generating snapshots.

#### Snapshot patch files​

When updating snapshots, Playwright creates patch files that capture differences. These patch files can be reviewed, applied, and committed to source control, allowing teams to track structural changes over time and ensure updates are consistent with application requirements.

### 3\. Using the `Locator.ariaSnapshot` method​

The [locator.ariaSnapshot()](/docs/api/class-locator#locator-aria-snapshot) method allows you to programmatically create a YAML representation of accessible elements within a locator’s scope, especially helpful for generating snapshots dynamically during test execution.

**Example** :
    
    
    const snapshot = await page.locator('body').ariaSnapshot();  
    console.log(snapshot);  
    

This command outputs the aria snapshot within the specified locator’s scope in YAML format, which you can validate or store as needed.

## Accessibility tree examples​

### Headings with level attributes​

Headings can include a `level` attribute indicating their heading level.
    
    
    <h1>Title</h1>  
    <h2>Subtitle</h2>  
    

_aria snapshot_
    
    
    - heading "Title" [level=1]  
    - heading "Subtitle" [level=2]  
    

### Text nodes​

Standalone or descriptive text elements appear as text nodes.
    
    
    <div>Sample accessible name</div>  
    

_aria snapshot_
    
    
    - text: Sample accessible name  
    

### Inline multiline text​

Multiline text, such as paragraphs, is normalized in the aria snapshot.
    
    
    <p>Line 1<br>Line 2</p>  
    

_aria snapshot_
    
    
    - paragraph: Line 1 Line 2  
    

### Links​

Links display their text or composed content from pseudo-elements.
    
    
    <a href="#more-info">Read more about Accessibility</a>  
    

_aria snapshot_
    
    
    - link "Read more about Accessibility"  
    

### Text boxes​

Input elements of type `text` show their `value` attribute content.
    
    
    <input type="text" value="Enter your name">  
    

_aria snapshot_
    
    
    - textbox: Enter your name  
    

### Lists with items​

Ordered and unordered lists include their list items.
    
    
    <ul aria-label="Main Features">  
      <li>Feature 1</li>  
      <li>Feature 2</li>  
    </ul>  
    

_aria snapshot_
    
    
    - list "Main Features":  
      - listitem: Feature 1  
      - listitem: Feature 2  
    

### Grouped elements​

Groups capture nested elements, such as `<details>` elements with summary content.
    
    
    <details>  
      <summary>Summary</summary>  
      <p>Detail content here</p>  
    </details>  
    

_aria snapshot_
    
    
    - group: Summary  
    

### Attributes and states​

Commonly used ARIA attributes, like `checked`, `disabled`, `expanded`, `level`, `pressed`, and `selected`, represent control states.

#### Checkbox with `checked` attribute​
    
    
    <input type="checkbox" checked>  
    

_aria snapshot_
    
    
    - checkbox [checked]  
    

#### Button with `pressed` attribute​
    
    
    <button aria-pressed="true">Toggle</button>  
    

_aria snapshot_
    
    
    - button "Toggle" [pressed=true]  
    
