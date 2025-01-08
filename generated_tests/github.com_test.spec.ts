import { test, expect, Page } from '@playwright/test';

test.describe('GitHub Homepage Tests', () => {
  let page: Page;

  test.beforeEach(async ({ browser }) => {
    page = await browser.newPage();
    await page.goto('https://github.com');
    await page.waitForLoadState('networkidle');
  });

  test.afterEach(async () => {
    await page?.close();
  });

  test('Basic Page Elements', async () => {
    // Verify page load
    await expect(page).toHaveURL(/.*github.com/);

    // Verify key elements
    await expect(page.getByRole('heading', { name: /Build.*software/i })).toBeVisible();
    await expect(page.getByRole('button', { name: /Try.*Copilot/i })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Sign up for GitHub' })).toBeVisible();
  });

  test('Navigation - ', async () => {
    // Verify navigation links are visible and clickable
    await expect(page.getByRole('link', { name: 'GitHub Copilot Write better code with AI' })).toBeVisible();
    await expect(page.getByRole('link', { name: 'Security Find and fix vulnerabilities' })).toBeVisible();
    await expect(page.getByRole('link', { name: 'Actions Automate any workflow' })).toBeVisible();
    await expect(page.getByRole('link', { name: 'Codespaces Instant dev environments' })).toBeVisible();
    await expect(page.getByRole('link', { name: 'Issues Plan and track work' })).toBeVisible();
    await expect(page.getByRole('link', { name: 'Code Review Manage code changes' })).toBeVisible();
    await expect(page.getByRole('link', { name: 'Discussions Collaborate outside of code' })).toBeVisible();
    await expect(page.getByRole('link', { name: 'Code Search Find more, search less' })).toBeVisible();
    await expect(page.getByRole('link', { name: 'Documentation' })).toBeVisible();
    await expect(page.getByRole('link', { name: 'GitHub Skills' })).toBeVisible();
    await expect(page.getByRole('link', { name: 'Blog' })).toBeVisible();
    await expect(page.getByRole('link', { name: 'View all solutions' })).toBeVisible();
    await expect(page.getByRole('link', { name: 'Learning Pathways' })).toBeVisible();
    await expect(page.getByRole('link', { name: 'White papers, Ebooks, Webinars' })).toBeVisible();
    await expect(page.getByRole('link', { name: 'Partners' })).toBeVisible();
    await expect(page.getByRole('link', { name: 'Enterprise platform AI-powered developer platform' })).toBeVisible();
    await expect(page.getByRole('link', { name: 'Advanced Security Enterprise-grade security features' })).toBeVisible();
    await expect(page.getByRole('link', { name: 'GitHub Copilot Enterprise-grade AI features' })).toBeVisible();
    await expect(page.getByRole('link', { name: 'Premium Support Enterprise-grade 24/7 support' })).toBeVisible();
    await expect(page.getByRole('link', { name: 'Pricing' })).toBeVisible();
    await expect(page.getByRole('link', { name: 'GitHub on LinkedIn' })).toBeVisible();
    await expect(page.getByRole('link', { name: 'Instagram GitHub on Instagram' })).toBeVisible();
    await expect(page.getByRole('link', { name: 'GitHub on YouTube' })).toBeVisible();
    await expect(page.getByRole('link', { name: 'GitHub on X' })).toBeVisible();
    await expect(page.getByRole('link', { name: 'TikTok GitHub on TikTok' })).toBeVisible();
    await expect(page.getByRole('link', { name: 'Twitch GitHub on Twitch' })).toBeVisible();
    await expect(page.getByRole('link', { name: 'GitHub’s organization on GitHub' })).toBeVisible();
  });

  test('Content Assertions', async () => {
    // Verify page content
    // Verify header: Site-wide Links
    await expect(page.getByRole('heading', { name: 'Site-wide Links' })).toBeVisible();

  });

  test('Error Handling', async () => {
    // Test invalid search
    const searchInput = page.getByPlaceholder(/Search/i);
    await searchInput.fill('!@#$%^&*()');
    await searchInput.press('Enter');
    await expect(page.getByText(/No results|no matches/i)).toBeVisible();

    // Test invalid navigation
    await page.goto('https://github.com/nonexistent-page');
    await expect(page.getByText(/Page not found|404/i)).toBeVisible();
  });
});