
import { test, expect } from '@playwright/test';

test.describe('https://github.com/features/copilot', () => {
    test.beforeEach(async ({ page }) => {
        await page.goto('https://github.com/features/copilot');
    });
    
    
    test('should handle button - Product', async ({ page }) => {
                await expect(page.getByText('Product')).toBeVisible();
        await page.getByText('Product').click();
    });
    

    test('should handle button - Solutions', async ({ page }) => {
                await expect(page.getByText('Solutions')).toBeVisible();
        await page.getByText('Solutions').click();
    });
    

    test('should handle button - Resources', async ({ page }) => {
                await expect(page.getByText('Resources')).toBeVisible();
        await page.getByText('Resources').click();
    });
    

    test('should handle button - Open Source', async ({ page }) => {
                await expect(page.getByText('Open Source')).toBeVisible();
        await page.getByText('Open Source').click();
    });
    

    test('should handle button - Enterprise', async ({ page }) => {
                await expect(page.getByText('Enterprise')).toBeVisible();
        await page.getByText('Enterprise').click();
    });
    

    test('should handle button - Search or jump to…', async ({ page }) => {
                await expect(page.getByRole('generic', { name: 'Search or jump to…' })).toBeVisible();
        await page.getByRole('generic', { name: 'Search or jump to…' }).click();
    });
    

    test('should handle button - GitHub Copilot Chat demo video', async ({ page }) => {
                await expect(page.getByRole('generic', { name: 'GitHub Copilot Chat demo video is currently playing. Click to pause.' })).toBeVisible();
        await page.getByRole('generic', { name: 'GitHub Copilot Chat demo video is currently playing. Click to pause.' }).click();
    });
    

    test('should handle button - Logo suite animation is curren', async ({ page }) => {
                await expect(page.getByRole('generic', { name: 'Logo suite animation is currently playing. Click to pause.' })).toBeVisible();
        await page.getByRole('generic', { name: 'Logo suite animation is currently playing. Click to pause.' }).click();
    });
    

    test('should handle button - Video demonstrating how GitHub', async ({ page }) => {
                await expect(page.getByRole('generic', { name: 'Video demonstrating how GitHub Copilot accelerates workflow through interactive codebase chat is currently paused. Click to play.' })).toBeVisible();
        await page.getByRole('generic', { name: 'Video demonstrating how GitHub Copilot accelerates workflow through interactive codebase chat is currently paused. Click to play.' }).click();
    });
    

    test('should handle button - GitHub Copilot model choices i', async ({ page }) => {
                await expect(page.getByRole('generic', { name: 'GitHub Copilot model choices is currently paused. Click to play.' })).toBeVisible();
        await page.getByRole('generic', { name: 'GitHub Copilot model choices is currently paused. Click to play.' }).click();
    });
    

    test('should handle button - General', async ({ page }) => {
                await expect(page.getByTestId('FAQGroup-tab-1')).toBeVisible();
        await page.getByTestId('FAQGroup-tab-1').click();
    });
    

    test('should handle button - Plans & Pricing', async ({ page }) => {
                await expect(page.getByTestId('FAQGroup-tab-2')).toBeVisible();
        await page.getByTestId('FAQGroup-tab-2').click();
    });
    

    test('should handle button - Privacy', async ({ page }) => {
                await expect(page.getByTestId('FAQGroup-tab-3')).toBeVisible();
        await page.getByTestId('FAQGroup-tab-3').click();
    });
    

    test('should handle button - Responsible AI', async ({ page }) => {
                await expect(page.getByTestId('FAQGroup-tab-4')).toBeVisible();
        await page.getByTestId('FAQGroup-tab-4').click();
    });
    

    test('should handle button - Manage cookies', async ({ page }) => {
                await expect(page.getByText('Manage cookies')).toBeVisible();
        await page.getByText('Manage cookies').click();
    });
    

    test('should handle button - Do not share my personal infor', async ({ page }) => {
                await expect(page.getByText('Do not share my personal information')).toBeVisible();
        await page.getByText('Do not share my personal information').click();
    });
    

    test('should handle a - Skip to content', async ({ page }) => {
                await expect(page.getByText('Skip to content')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Skip to content').click()
        ]);
        await expect(page).toHaveURL('#start-of-content');
    });
    

    test('should handle a - Homepage', async ({ page }) => {
                await expect(page.getByRole('generic', { name: 'Homepage' })).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByRole('generic', { name: 'Homepage' }).click()
        ]);
        await expect(page).toHaveURL('/');
    });
    

    test('should handle a - Pricing', async ({ page }) => {
                await expect(page.getByText('Pricing')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Pricing').click()
        ]);
        await expect(page).toHaveURL('https://github.com/pricing');
    });
    

    test('should handle a - Sign in', async ({ page }) => {
                await expect(page.getByText('Sign in')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Sign in').click()
        ]);
        await expect(page).toHaveURL('/login?return_to=https%3A%2F%2Fgithub.com%2Ffeatures%2Fcopilot');
    });
    

    test('should handle a - Sign up', async ({ page }) => {
                await expect(page.getByText('Sign up')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Sign up').click()
        ]);
        await expect(page).toHaveURL('/signup?ref_cta=Sign+up&ref_loc=header+logged+out&ref_page=%2Ffeatures%2Fcopilot&source=header');
    });
    

    test('should handle a - GitHub Copilot', async ({ page }) => {
                await expect(page.getByTestId('SubNav-root-heading')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByTestId('SubNav-root-heading').click()
        ]);
        await expect(page).toHaveURL('/features/copilot');
    });
    

    test('should handle a - Plans', async ({ page }) => {
                await expect(page.getByTestId('SubNav-root-link')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByTestId('SubNav-root-link').click()
        ]);
        await expect(page).toHaveURL('/features/copilot/plans');
    });
    

    test('should handle a - Tutorials', async ({ page }) => {
                await expect(page.getByTestId('SubNav-root-link')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByTestId('SubNav-root-link').click()
        ]);
        await expect(page).toHaveURL('/features/copilot/tutorials');
    });
    

    test('should handle a - Extensions', async ({ page }) => {
                await expect(page.getByTestId('SubNav-root-link')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByTestId('SubNav-root-link').click()
        ]);
        await expect(page).toHaveURL('/features/copilot/extensions');
    });
    

    test('should handle a - What’s new', async ({ page }) => {
                await expect(page.getByTestId('SubNav-root-link')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByTestId('SubNav-root-link').click()
        ]);
        await expect(page).toHaveURL('/features/copilot/whats-new');
    });
    

    test('should handle a - Get started for free', async ({ page }) => {
                await expect(page.getByText('Get started for free')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Get started for free').click()
        ]);
        await expect(page).toHaveURL('https://github.com/copilot');
    });
    

    test('should handle a - See plans & pricing', async ({ page }) => {
                await expect(page.getByText('See plans & pricing')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('See plans & pricing').click()
        ]);
        await expect(page).toHaveURL('/features/copilot/plans?cft=copilot_lo.features_copilot');
    });
    

    test('should handle a - Open now', async ({ page }) => {
                await expect(page.getByText('Open now')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Open now').click()
        ]);
        await expect(page).toHaveURL('vscode://GitHub.copilot');
    });
    

    test('should handle a - Our favorite Copilot prompts', async ({ page }) => {
                await expect(page.getByText('Our favorite Copilot prompts')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Our favorite Copilot prompts').click()
        ]);
        await expect(page).toHaveURL('https://docs.github.com/en/copilot/example-prompts-for-github-copilot-chat');
    });
    

    test('should handle a - Claude 3.5 Sonnet', async ({ page }) => {
                await expect(page.getByText('Claude 3.5 Sonnet')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Claude 3.5 Sonnet').click()
        ]);
        await expect(page).toHaveURL('https://docs.github.com/en/copilot/using-github-copilot/using-claude-sonnet-in-github-copilot');
    });
    

    test('should handle a - unknown', async ({ page }) => {
                await expect(page.locator('a')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.locator('a').click()
        ]);
        await expect(page).toHaveURL('https://docs.github.com/en/copilot/using-github-copilot/asking-github-copilot-questions-in-your-ide#slash-commands');
    });
    

    test('should handle a - Use slash commands', async ({ page }) => {
                await expect(page.getByText('Use slash commands')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Use slash commands').click()
        ]);
        await expect(page).toHaveURL('https://docs.github.com/en/copilot/using-github-copilot/asking-github-copilot-questions-in-your-ide#slash-commands');
    });
    

    test('should handle a - unknown', async ({ page }) => {
                await expect(page.locator('a')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.locator('a').click()
        ]);
        await expect(page).toHaveURL('https://docs.github.com/en/copilot/quickstart#introduction');
    });
    

    test('should handle a - Get early access', async ({ page }) => {
                await expect(page.getByText('Get early access')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Get early access').click()
        ]);
        await expect(page).toHaveURL('https://github.com/features/preview/copilot-code-review');
    });
    

    test('should handle a - unknown', async ({ page }) => {
                await expect(page.locator('a')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.locator('a').click()
        ]);
        await expect(page).toHaveURL('https://github.com/features/preview/copilot-code-review');
    });
    

    test('should handle a - Explore GitHub Copilot Extensi', async ({ page }) => {
                await expect(page.getByText('Explore GitHub Copilot Extensions')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Explore GitHub Copilot Extensions').click()
        ]);
        await expect(page).toHaveURL('https://github.com/features/copilot/extensions');
    });
    

    test('should handle a - unknown', async ({ page }) => {
                await expect(page.locator('a')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.locator('a').click()
        ]);
        await expect(page).toHaveURL('https://github.com/features/copilot/extensions');
    });
    

    test('should handle a - Try Copilot in the CLI', async ({ page }) => {
                await expect(page.getByText('Try Copilot in the CLI')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Try Copilot in the CLI').click()
        ]);
        await expect(page).toHaveURL('https://docs.github.com/copilot/using-github-copilot/using-github-copilot-in-the-command-line');
    });
    

    test('should handle a - unknown', async ({ page }) => {
                await expect(page.locator('a')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.locator('a').click()
        ]);
        await expect(page).toHaveURL('https://play.google.com/store/apps/details?id=com.github.android');
    });
    

    test('should handle a - unknown', async ({ page }) => {
                await expect(page.locator('a')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.locator('a').click()
        ]);
        await expect(page).toHaveURL('https://apps.apple.com/app/github/id1477376905?ls=1');
    });
    

    test('should handle a - Get started', async ({ page }) => {
                await expect(page.getByText('Get started')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Get started').click()
        ]);
        await expect(page).toHaveURL('https://github.com/copilot');
    });
    

    test('should handle a - Open now', async ({ page }) => {
                await expect(page.getByText('Open now')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Open now').click()
        ]);
        await expect(page).toHaveURL('vscode://GitHub.copilot');
    });
    

    test('should handle a - Get started', async ({ page }) => {
                await expect(page.getByText('Get started')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Get started').click()
        ]);
        await expect(page).toHaveURL('/login?return_to=%2Fgithub-copilot%2Fsignup%3Fcft%3Dcopilot_lo.features_copilot.cfi&cft=copilot_lo.features_copilot.cfi');
    });
    

    test('should handle a - Learn more', async ({ page }) => {
                await expect(page.getByText('Learn more')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Learn more').click()
        ]);
        await expect(page).toHaveURL('https://github.com/education');
    });
    

    test('should handle a - Get started', async ({ page }) => {
                await expect(page.getByText('Get started')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Get started').click()
        ]);
        await expect(page).toHaveURL('/login?return_to=%2Fgithub-copilot%2Fbusiness_signup%3Fcft%3Dcopilot_lo.features_copilot.cfb&cft=copilot_lo.features_copilot.cfb');
    });
    

    test('should handle a - Contact sales', async ({ page }) => {
                await expect(page.getByText('Contact sales')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Contact sales').click()
        ]);
        await expect(page).toHaveURL('/enterprise/contact?ref_cta=Contact+sales&ref_loc=pricing&ref_page=%2Ffeatures%2Fcopilot&scid=&utm_campaign=Copilot_feature_page_contact_sales_cta_utmroutercampaign&utm_medium=site&utm_source=github&cft=copilot_lo.features_copilot&utm_content=CopilotBusiness');
    });
    

    test('should handle a - Contact sales', async ({ page }) => {
                await expect(page.getByText('Contact sales')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Contact sales').click()
        ]);
        await expect(page).toHaveURL('/enterprise/contact?ref_cta=Contact+sales&ref_loc=pricing&ref_page=%2Ffeatures%2Fcopilot&scid=&utm_campaign=Copilot_feature_page_contact_sales_cta_utmroutercampaign&utm_medium=site&utm_source=github&cft=copilot_lo.features_copilot&utm_content=CopilotEnterprise');
    });
    

    test('should handle a - GitHub', async ({ page }) => {
                await expect(page.getByText('GitHub')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('GitHub').click()
        ]);
        await expect(page).toHaveURL('https://github.com/copilot');
    });
    

    test('should handle a - VS Code', async ({ page }) => {
                await expect(page.getByText('VS Code')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('VS Code').click()
        ]);
        await expect(page).toHaveURL('https://marketplace.visualstudio.com/items?itemName=GitHub.copilot');
    });
    

    test('should handle a - Visual Studio', async ({ page }) => {
                await expect(page.getByText('Visual Studio')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Visual Studio').click()
        ]);
        await expect(page).toHaveURL('https://visualstudio.microsoft.com/github-copilot');
    });
    

    test('should handle a - Xcode', async ({ page }) => {
                await expect(page.getByText('Xcode')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Xcode').click()
        ]);
        await expect(page).toHaveURL('https://github.com/github/CopilotForXcode');
    });
    

    test('should handle a - JetBrains IDEs', async ({ page }) => {
                await expect(page.getByText('JetBrains IDEs')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('JetBrains IDEs').click()
        ]);
        await expect(page).toHaveURL('https://plugins.jetbrains.com/plugin/17718-github-copilot');
    });
    

    test('should handle a - Neovim', async ({ page }) => {
                await expect(page.getByText('Neovim')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Neovim').click()
        ]);
        await expect(page).toHaveURL('https://github.com/github/copilot.vim');
    });
    

    test('should handle a - Azure Data Studio', async ({ page }) => {
                await expect(page.getByText('Azure Data Studio')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Azure Data Studio').click()
        ]);
        await expect(page).toHaveURL('https://learn.microsoft.com/en-us/azure-data-studio/extensions/github-copilot-extension-overview');
    });
    

    test('should handle a - See all supported editors', async ({ page }) => {
                await expect(page.getByText('See all supported editors')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('See all supported editors').click()
        ]);
        await expect(page).toHaveURL('https://docs.github.com/en/copilot/managing-copilot/configure-personal-settings/installing-the-github-copilot-extension-in-your-environment');
    });
    

    test('should handle a - 1', async ({ page }) => {
                await expect(page.locator('#footnote-ref-1-1')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.locator('#footnote-ref-1-1').click()
        ]);
        await expect(page).toHaveURL('#footnote-1');
    });
    

    test('should handle a - Get started', async ({ page }) => {
                await expect(page.getByText('Get started')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Get started').click()
        ]);
        await expect(page).toHaveURL('https://github.com/copilot');
    });
    

    test('should handle a - Get started', async ({ page }) => {
                await expect(page.getByText('Get started')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Get started').click()
        ]);
        await expect(page).toHaveURL('/login?return_to=%2Fgithub-copilot%2Fsignup%3Fcft%3Dcopilot_lo.features_copilot.cfi&cft=copilot_lo.features_copilot.cfi');
    });
    

    test('should handle a - Get started', async ({ page }) => {
                await expect(page.getByText('Get started')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Get started').click()
        ]);
        await expect(page).toHaveURL('/login?return_to=%2Fgithub-copilot%2Fbusiness_signup%3Fcft%3Dcopilot_lo.features_copilot.cfb&cft=copilot_lo.features_copilot.cfb');
    });
    

    test('should handle a - Contact sales', async ({ page }) => {
                await expect(page.getByText('Contact sales')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Contact sales').click()
        ]);
        await expect(page).toHaveURL('/enterprise/contact?ref_cta=Contact+sales&ref_loc=pricing&ref_page=%2Ffeatures%2Fcopilot&scid=&utm_campaign=Copilot_feature_page_contact_sales_cta_utmroutercampaign&utm_medium=site&utm_source=github&cft=copilot_lo.features_copilot&utm_content=CopilotEnterprise');
    });
    

    test('should handle a - Preview the latest features', async ({ page }) => {
                await expect(page.getByText('Preview the latest features')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Preview the latest features').click()
        ]);
        await expect(page).toHaveURL('https://github.com/features/preview');
    });
    

    test('should handle a - Explore The GitHub Blog', async ({ page }) => {
                await expect(page.getByText('Explore The GitHub Blog')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Explore The GitHub Blog').click()
        ]);
        await expect(page).toHaveURL('https://github.blog/');
    });
    

    test('should handle a - Visit the GitHub Copilot Trust', async ({ page }) => {
                await expect(page.getByText('Visit the GitHub Copilot Trust Center')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Visit the GitHub Copilot Trust Center').click()
        ]);
        await expect(page).toHaveURL('https://copilot.github.trust.page/');
    });
    

    test('should handle a - Authentication with SAML singl', async ({ page }) => {
                await expect(page.getByText('Authentication with SAML single sign-on (SSO)')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Authentication with SAML single sign-on (SSO)').click()
        ]);
        await expect(page).toHaveURL('https://docs.github.com/en/enterprise-cloud@latest/authentication/authenticating-with-saml-single-sign-on/about-authentication-with-saml-single-sign-on');
    });
    

    test('should handle a - Back to content', async ({ page }) => {
                await expect(page.getByRole('generic', { name: 'Back to content' })).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByRole('generic', { name: 'Back to content' }).click()
        ]);
        await expect(page).toHaveURL('#footnote-ref-1-1');
    });
    

    test('should handle a - Go to GitHub homepage', async ({ page }) => {
                await expect(page.getByRole('generic', { name: 'Go to GitHub homepage' })).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByRole('generic', { name: 'Go to GitHub homepage' }).click()
        ]);
        await expect(page).toHaveURL('/');
    });
    

    test('should handle a - Subscribe', async ({ page }) => {
                await expect(page.getByText('Subscribe')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Subscribe').click()
        ]);
        await expect(page).toHaveURL('https://resources.github.com/newsletter/');
    });
    

    test('should handle a - Features', async ({ page }) => {
                await expect(page.getByText('Features')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Features').click()
        ]);
        await expect(page).toHaveURL('/features');
    });
    

    test('should handle a - Enterprise', async ({ page }) => {
                await expect(page.getByText('Enterprise')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Enterprise').click()
        ]);
        await expect(page).toHaveURL('/enterprise');
    });
    

    test('should handle a - Copilot', async ({ page }) => {
                await expect(page.getByText('Copilot')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Copilot').click()
        ]);
        await expect(page).toHaveURL('/features/copilot');
    });
    

    test('should handle a - Security', async ({ page }) => {
                await expect(page.getByText('Security')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Security').click()
        ]);
        await expect(page).toHaveURL('/security');
    });
    

    test('should handle a - Pricing', async ({ page }) => {
                await expect(page.getByText('Pricing')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Pricing').click()
        ]);
        await expect(page).toHaveURL('/pricing');
    });
    

    test('should handle a - Team', async ({ page }) => {
                await expect(page.getByText('Team')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Team').click()
        ]);
        await expect(page).toHaveURL('/team');
    });
    

    test('should handle a - Resources', async ({ page }) => {
                await expect(page.getByText('Resources')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Resources').click()
        ]);
        await expect(page).toHaveURL('https://resources.github.com');
    });
    

    test('should handle a - Roadmap', async ({ page }) => {
                await expect(page.getByText('Roadmap')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Roadmap').click()
        ]);
        await expect(page).toHaveURL('https://github.com/github/roadmap');
    });
    

    test('should handle a - Compare GitHub', async ({ page }) => {
                await expect(page.getByText('Compare GitHub')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Compare GitHub').click()
        ]);
        await expect(page).toHaveURL('https://resources.github.com/devops/tools/compare');
    });
    

    test('should handle a - Developer API', async ({ page }) => {
                await expect(page.getByText('Developer API')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Developer API').click()
        ]);
        await expect(page).toHaveURL('https://docs.github.com/get-started/exploring-integrations/about-building-integrations');
    });
    

    test('should handle a - Partners', async ({ page }) => {
                await expect(page.getByText('Partners')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Partners').click()
        ]);
        await expect(page).toHaveURL('https://partner.github.com');
    });
    

    test('should handle a - Education', async ({ page }) => {
                await expect(page.getByText('Education')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Education').click()
        ]);
        await expect(page).toHaveURL('https://github.com/edu');
    });
    

    test('should handle a - GitHub CLI', async ({ page }) => {
                await expect(page.getByText('GitHub CLI')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('GitHub CLI').click()
        ]);
        await expect(page).toHaveURL('https://cli.github.com');
    });
    

    test('should handle a - GitHub Desktop', async ({ page }) => {
                await expect(page.getByText('GitHub Desktop')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('GitHub Desktop').click()
        ]);
        await expect(page).toHaveURL('https://desktop.github.com');
    });
    

    test('should handle a - GitHub Mobile', async ({ page }) => {
                await expect(page.getByText('GitHub Mobile')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('GitHub Mobile').click()
        ]);
        await expect(page).toHaveURL('https://github.com/mobile');
    });
    

    test('should handle a - Docs', async ({ page }) => {
                await expect(page.getByText('Docs')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Docs').click()
        ]);
        await expect(page).toHaveURL('https://docs.github.com');
    });
    

    test('should handle a - Community Forum', async ({ page }) => {
                await expect(page.getByText('Community Forum')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Community Forum').click()
        ]);
        await expect(page).toHaveURL('https://github.community');
    });
    

    test('should handle a - Professional Services', async ({ page }) => {
                await expect(page.getByText('Professional Services')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Professional Services').click()
        ]);
        await expect(page).toHaveURL('https://services.github.com');
    });
    

    test('should handle a - Premium Support', async ({ page }) => {
                await expect(page.getByText('Premium Support')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Premium Support').click()
        ]);
        await expect(page).toHaveURL('/enterprise/premium-support');
    });
    

    test('should handle a - Skills', async ({ page }) => {
                await expect(page.getByText('Skills')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Skills').click()
        ]);
        await expect(page).toHaveURL('https://skills.github.com');
    });
    

    test('should handle a - Status', async ({ page }) => {
                await expect(page.getByText('Status')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Status').click()
        ]);
        await expect(page).toHaveURL('https://www.githubstatus.com');
    });
    

    test('should handle a - Contact GitHub', async ({ page }) => {
                await expect(page.getByText('Contact GitHub')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Contact GitHub').click()
        ]);
        await expect(page).toHaveURL('https://support.github.com?tags=dotcom-footer');
    });
    

    test('should handle a - About', async ({ page }) => {
                await expect(page.getByText('About')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('About').click()
        ]);
        await expect(page).toHaveURL('https://github.com/about');
    });
    

    test('should handle a - Customer stories', async ({ page }) => {
                await expect(page.getByText('Customer stories')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Customer stories').click()
        ]);
        await expect(page).toHaveURL('/customer-stories?type=enterprise');
    });
    

    test('should handle a - Blog', async ({ page }) => {
                await expect(page.getByText('Blog')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Blog').click()
        ]);
        await expect(page).toHaveURL('https://github.blog');
    });
    

    test('should handle a - The ReadME Project', async ({ page }) => {
                await expect(page.getByText('The ReadME Project')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('The ReadME Project').click()
        ]);
        await expect(page).toHaveURL('/readme');
    });
    

    test('should handle a - Careers', async ({ page }) => {
                await expect(page.getByText('Careers')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Careers').click()
        ]);
        await expect(page).toHaveURL('https://github.careers');
    });
    

    test('should handle a - Newsroom', async ({ page }) => {
                await expect(page.getByText('Newsroom')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Newsroom').click()
        ]);
        await expect(page).toHaveURL('/newsroom');
    });
    

    test('should handle a - Inclusion', async ({ page }) => {
                await expect(page.getByText('Inclusion')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Inclusion').click()
        ]);
        await expect(page).toHaveURL('/about/diversity');
    });
    

    test('should handle a - Social Impact', async ({ page }) => {
                await expect(page.getByText('Social Impact')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Social Impact').click()
        ]);
        await expect(page).toHaveURL('https://socialimpact.github.com');
    });
    

    test('should handle a - Shop', async ({ page }) => {
                await expect(page.getByText('Shop')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Shop').click()
        ]);
        await expect(page).toHaveURL('https://shop.github.com');
    });
    

    test('should handle a - Terms', async ({ page }) => {
                await expect(page.getByText('Terms')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Terms').click()
        ]);
        await expect(page).toHaveURL('https://docs.github.com/site-policy/github-terms/github-terms-of-service');
    });
    

    test('should handle a - Privacy', async ({ page }) => {
                await expect(page.getByText('Privacy')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Privacy').click()
        ]);
        await expect(page).toHaveURL('https://docs.github.com/site-policy/privacy-policies/github-privacy-statement');
    });
    

    test('should handle a - Updated 02/2024', async ({ page }) => {
                await expect(page.getByText('Updated 02/2024')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Updated 02/2024').click()
        ]);
        await expect(page).toHaveURL('https://github.com/github/site-policy/pull/582');
    });
    

    test('should handle a - Sitemap', async ({ page }) => {
                await expect(page.getByText('Sitemap')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Sitemap').click()
        ]);
        await expect(page).toHaveURL('/sitemap');
    });
    

    test('should handle a - What is Git?', async ({ page }) => {
                await expect(page.getByText('What is Git?')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('What is Git?').click()
        ]);
        await expect(page).toHaveURL('/git-guides');
    });
    

    test('should handle a - GitHub on LinkedIn', async ({ page }) => {
                await expect(page.getByText('GitHub on LinkedIn')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('GitHub on LinkedIn').click()
        ]);
        await expect(page).toHaveURL('https://www.linkedin.com/company/github');
    });
    

    test('should handle a - Instagram

        GitHub on I', async ({ page }) => {
                await expect(page.getByText('Instagram

        GitHub on Instagram')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Instagram

        GitHub on Instagram').click()
        ]);
        await expect(page).toHaveURL('https://www.instagram.com/github');
    });
    

    test('should handle a - GitHub on YouTube', async ({ page }) => {
                await expect(page.getByText('GitHub on YouTube')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('GitHub on YouTube').click()
        ]);
        await expect(page).toHaveURL('https://www.youtube.com/github');
    });
    

    test('should handle a - GitHub on X', async ({ page }) => {
                await expect(page.getByText('GitHub on X')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('GitHub on X').click()
        ]);
        await expect(page).toHaveURL('https://x.com/github');
    });
    

    test('should handle a - TikTok

        GitHub on TikT', async ({ page }) => {
                await expect(page.getByText('TikTok

        GitHub on TikTok')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('TikTok

        GitHub on TikTok').click()
        ]);
        await expect(page).toHaveURL('https://www.tiktok.com/@github');
    });
    

    test('should handle a - Twitch

        GitHub on Twit', async ({ page }) => {
                await expect(page.getByText('Twitch

        GitHub on Twitch')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Twitch

        GitHub on Twitch').click()
        ]);
        await expect(page).toHaveURL('https://www.twitch.tv/github');
    });
    

    test('should handle a - GitHub’s organization on GitHu', async ({ page }) => {
                await expect(page.getByText('GitHub’s organization on GitHub')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('GitHub’s organization on GitHub').click()
        ]);
        await expect(page).toHaveURL('https://github.com/github');
    });
    

    test('should handle nav - GitHub CopilotOverviewPlansTut', async ({ page }) => {
                await expect(page.getByTestId('SubNav-root')).toBeVisible();
    });
    

    test('should handle a - GitHub Copilot', async ({ page }) => {
                await expect(page.getByTestId('SubNav-root-heading')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByTestId('SubNav-root-heading').click()
        ]);
        await expect(page).toHaveURL('/features/copilot');
    });
    

    test('should handle ul - OverviewPlansTutorialsExtensio', async ({ page }) => {
                await expect(page.getByTestId('SubNav-root-overlay')).toBeVisible();
    });
    

    test('should handle a - Plans', async ({ page }) => {
                await expect(page.getByTestId('SubNav-root-link')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByTestId('SubNav-root-link').click()
        ]);
        await expect(page).toHaveURL('/features/copilot/plans');
    });
    

    test('should handle a - Tutorials', async ({ page }) => {
                await expect(page.getByTestId('SubNav-root-link')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByTestId('SubNav-root-link').click()
        ]);
        await expect(page).toHaveURL('/features/copilot/tutorials');
    });
    

    test('should handle a - Extensions', async ({ page }) => {
                await expect(page.getByTestId('SubNav-root-link')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByTestId('SubNav-root-link').click()
        ]);
        await expect(page).toHaveURL('/features/copilot/extensions');
    });
    

    test('should handle a - What’s new', async ({ page }) => {
                await expect(page.getByTestId('SubNav-root-link')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByTestId('SubNav-root-link').click()
        ]);
        await expect(page).toHaveURL('/features/copilot/whats-new');
    });
    

    test('should handle div - GitHub Copilot is now availabl', async ({ page }) => {
                await expect(page.getByTestId('Grid-:R1pb:')).toBeVisible();
    });
    

    test('should handle div - GitHub Copilot is now availabl', async ({ page }) => {
                await expect(page.getByTestId('Grid-:Rdpb:')).toBeVisible();
    });
    

    test('should handle span - GitHub Copilot is now availabl', async ({ page }) => {
                await expect(page.getByTestId('Label')).toBeVisible();
    });
    

    test('should handle span - unknown', async ({ page }) => {
                await expect(page.getByTestId('Label-leading-visual')).toBeVisible();
    });
    

    test('should handle div - unknown', async ({ page }) => {
                await expect(page.getByTestId('LogoSuite-marqueeGroup')).toBeVisible();
    });
    

    test('should handle div - FeaturesCustomizable. Contextu', async ({ page }) => {
                await expect(page.getByTestId('Grid-:R2b:')).toBeVisible();
    });
    

    test('should handle span - Features', async ({ page }) => {
                await expect(page.getByTestId('Label')).toBeVisible();
    });
    

    test('should handle div - PricingTake flight with GitHub', async ({ page }) => {
                await expect(page.getByTestId('Grid-:Rbb:')).toBeVisible();
    });
    

    test('should handle span - Pricing', async ({ page }) => {
                await expect(page.getByTestId('Label')).toBeVisible();
    });
    

    test('should handle div - NewFreeFor developers looking ', async ({ page }) => {
                await expect(page.getByTestId('Grid-:Rjb:')).toBeVisible();
    });
    

    test('should handle div - NewFreeFor developers looking ', async ({ page }) => {
                await expect(page.getByTestId('Grid-:R2jb:')).toBeVisible();
    });
    

    test('should handle span - New', async ({ page }) => {
                await expect(page.getByTestId('Label')).toBeVisible();
    });
    

    test('should handle svg - unknown', async ({ page }) => {
                await expect(page.getByTestId('Button-expandable-arrow')).toBeVisible();
    });
    

    test('should handle span - Most popular', async ({ page }) => {
                await expect(page.getByTestId('Label')).toBeVisible();
    });
    

    test('should handle svg - unknown', async ({ page }) => {
                await expect(page.getByTestId('Button-expandable-arrow')).toBeVisible();
    });
    

    test('should handle svg - unknown', async ({ page }) => {
                await expect(page.getByTestId('Button-expandable-arrow')).toBeVisible();
    });
    

    test('should handle svg - unknown', async ({ page }) => {
                await expect(page.getByTestId('Button-expandable-arrow')).toBeVisible();
    });
    

    test('should handle svg - unknown', async ({ page }) => {
                await expect(page.getByTestId('Button-expandable-arrow')).toBeVisible();
    });
    

    test('should handle div - GitHub Copilot is available on', async ({ page }) => {
                await expect(page.getByTestId('Grid-:Rrb:')).toBeVisible();
    });
    

    test('should handle div - Compare featuresFeaturesFreePr', async ({ page }) => {
                await expect(page.getByTestId('Grid-:R13b:')).toBeVisible();
    });
    

    test('should handle span - Preview', async ({ page }) => {
                await expect(page.getByTestId('Label')).toBeVisible();
    });
    

    test('should handle span - Preview', async ({ page }) => {
                await expect(page.getByTestId('Label')).toBeVisible();
    });
    

    test('should handle span - Preview', async ({ page }) => {
                await expect(page.getByTestId('Label')).toBeVisible();
    });
    

    test('should handle span - Preview', async ({ page }) => {
                await expect(page.getByTestId('Label')).toBeVisible();
    });
    

    test('should handle span - Preview', async ({ page }) => {
                await expect(page.getByTestId('Label')).toBeVisible();
    });
    

    test('should handle span - Preview', async ({ page }) => {
                await expect(page.getByTestId('Label')).toBeVisible();
    });
    

    test('should handle span - Preview', async ({ page }) => {
                await expect(page.getByTestId('Label')).toBeVisible();
    });
    

    test('should handle span - Preview', async ({ page }) => {
                await expect(page.getByTestId('Label')).toBeVisible();
    });
    

    test('should handle span - Preview', async ({ page }) => {
                await expect(page.getByTestId('Label')).toBeVisible();
    });
    

    test('should handle span - Preview', async ({ page }) => {
                await expect(page.getByTestId('Label')).toBeVisible();
    });
    

    test('should handle span - Preview', async ({ page }) => {
                await expect(page.getByTestId('Label')).toBeVisible();
    });
    

    test('should handle svg - unknown', async ({ page }) => {
                await expect(page.getByTestId('Button-expandable-arrow')).toBeVisible();
    });
    

    test('should handle svg - unknown', async ({ page }) => {
                await expect(page.getByTestId('Button-expandable-arrow')).toBeVisible();
    });
    

    test('should handle svg - unknown', async ({ page }) => {
                await expect(page.getByTestId('Button-expandable-arrow')).toBeVisible();
    });
    

    test('should handle svg - unknown', async ({ page }) => {
                await expect(page.getByTestId('Button-expandable-arrow')).toBeVisible();
    });
    

    test('should handle div - Get the most out of GitHub Cop', async ({ page }) => {
                await expect(page.getByTestId('Grid-:R4b:')).toBeVisible();
    });
    

    test('should handle div - Frequently asked questionsGene', async ({ page }) => {
                await expect(page.getByTestId('Grid-:R5b:')).toBeVisible();
    });
    

    test('should handle div - Frequently asked questionsGene', async ({ page }) => {
                await expect(page.getByTestId('Grid-:R2tb:')).toBeVisible();
    });
    

    test('should handle div - Frequently asked questions', async ({ page }) => {
                await expect(page.getByTestId('Grid-:Retb:')).toBeVisible();
    });
    

    test('should handle div - GeneralPlans & PricingPrivacyR', async ({ page }) => {
                await expect(page.getByTestId('Grid-:Rutb:')).toBeVisible();
    });
    

    test('should handle button - General', async ({ page }) => {
                await expect(page.getByTestId('FAQGroup-tab-1')).toBeVisible();
        await page.getByTestId('FAQGroup-tab-1').click();
    });
    

    test('should handle button - Plans & Pricing', async ({ page }) => {
                await expect(page.getByTestId('FAQGroup-tab-2')).toBeVisible();
        await page.getByTestId('FAQGroup-tab-2').click();
    });
    

    test('should handle button - Privacy', async ({ page }) => {
                await expect(page.getByTestId('FAQGroup-tab-3')).toBeVisible();
        await page.getByTestId('FAQGroup-tab-3').click();
    });
    

    test('should handle button - Responsible AI', async ({ page }) => {
                await expect(page.getByTestId('FAQGroup-tab-4')).toBeVisible();
        await page.getByTestId('FAQGroup-tab-4').click();
    });
    

    test('should handle div - GeneralWhat is GitHub Copilot?', async ({ page }) => {
                await expect(page.getByTestId('FAQGroup-tab-panel-1')).toBeVisible();
    });
    

    test('should handle h3 - General', async ({ page }) => {
                await expect(page.getByTestId('FAQGroup-tab-panel-heading-1')).toBeVisible();
    });
    

    test('should handle div - FootnotesAuthentication with S', async ({ page }) => {
                await expect(page.getByTestId('Grid-:R6b:')).toBeVisible();
    });
    

    test('should handle a - Homepage', async ({ page }) => {
                await expect(page.getByRole('generic', { name: 'Homepage' })).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByRole('generic', { name: 'Homepage' }).click()
        ]);
        await expect(page).toHaveURL('/');
    });
    

    test('should handle nav - Global', async ({ page }) => {
                await expect(page.getByRole('generic', { name: 'Global' })).toBeVisible();
    });
    

    test('should handle button - Search or jump to…', async ({ page }) => {
                await expect(page.getByRole('generic', { name: 'Search or jump to…' })).toBeVisible();
        await page.getByRole('generic', { name: 'Search or jump to…' }).click();
    });
    

    test('should handle div - A demonstration animation of a', async ({ page }) => {
                await expect(page.getByRole('img', { name: 'A demonstration animation of a code editor using GitHub Copilot Chat, where the user requests GitHub Copilot to generate unit tests for a given code snippet.' })).toBeVisible();
    });
    

    test('should handle button - GitHub Copilot Chat demo video', async ({ page }) => {
                await expect(page.getByRole('generic', { name: 'GitHub Copilot Chat demo video is currently playing. Click to pause.' })).toBeVisible();
        await page.getByRole('generic', { name: 'GitHub Copilot Chat demo video is currently playing. Click to pause.' }).click();
    });
    

    test('should handle img - Lyft', async ({ page }) => {
                await expect(page.getByRole('generic', { name: 'Lyft' })).toBeVisible();
    });
    

    test('should handle img - FedEx', async ({ page }) => {
                await expect(page.getByRole('generic', { name: 'FedEx' })).toBeVisible();
    });
    

    test('should handle img - AT&T', async ({ page }) => {
                await expect(page.getByRole('generic', { name: 'AT&T' })).toBeVisible();
    });
    

    test('should handle img - Shopify', async ({ page }) => {
                await expect(page.getByRole('generic', { name: 'Shopify' })).toBeVisible();
    });
    

    test('should handle img - BMW', async ({ page }) => {
                await expect(page.getByRole('generic', { name: 'BMW' })).toBeVisible();
    });
    

    test('should handle img - H&M', async ({ page }) => {
                await expect(page.getByRole('generic', { name: 'H&M' })).toBeVisible();
    });
    

    test('should handle img - Honeywell', async ({ page }) => {
                await expect(page.getByRole('generic', { name: 'Honeywell' })).toBeVisible();
    });
    

    test('should handle img - EY', async ({ page }) => {
                await expect(page.getByRole('generic', { name: 'EY' })).toBeVisible();
    });
    

    test('should handle img - Infosys', async ({ page }) => {
                await expect(page.getByRole('generic', { name: 'Infosys' })).toBeVisible();
    });
    

    test('should handle img - BBVA', async ({ page }) => {
                await expect(page.getByRole('generic', { name: 'BBVA' })).toBeVisible();
    });
    

    test('should handle img - Lyft', async ({ page }) => {
                await expect(page.getByRole('generic', { name: 'Lyft' })).toBeVisible();
    });
    

    test('should handle img - FedEx', async ({ page }) => {
                await expect(page.getByRole('generic', { name: 'FedEx' })).toBeVisible();
    });
    

    test('should handle img - AT&T', async ({ page }) => {
                await expect(page.getByRole('generic', { name: 'AT&T' })).toBeVisible();
    });
    

    test('should handle img - Shopify', async ({ page }) => {
                await expect(page.getByRole('generic', { name: 'Shopify' })).toBeVisible();
    });
    

    test('should handle img - BMW', async ({ page }) => {
                await expect(page.getByRole('generic', { name: 'BMW' })).toBeVisible();
    });
    

    test('should handle img - H&M', async ({ page }) => {
                await expect(page.getByRole('generic', { name: 'H&M' })).toBeVisible();
    });
    

    test('should handle img - Honeywell', async ({ page }) => {
                await expect(page.getByRole('generic', { name: 'Honeywell' })).toBeVisible();
    });
    

    test('should handle img - EY', async ({ page }) => {
                await expect(page.getByRole('generic', { name: 'EY' })).toBeVisible();
    });
    

    test('should handle img - Infosys', async ({ page }) => {
                await expect(page.getByRole('generic', { name: 'Infosys' })).toBeVisible();
    });
    

    test('should handle img - BBVA', async ({ page }) => {
                await expect(page.getByRole('generic', { name: 'BBVA' })).toBeVisible();
    });
    

    test('should handle button - Logo suite animation is curren', async ({ page }) => {
                await expect(page.getByRole('generic', { name: 'Logo suite animation is currently playing. Click to pause.' })).toBeVisible();
        await page.getByRole('generic', { name: 'Logo suite animation is currently playing. Click to pause.' }).click();
    });
    

    test('should handle video - Video demonstrating how GitHub', async ({ page }) => {
                await expect(page.getByRole('generic', { name: 'Video demonstrating how GitHub Copilot accelerates workflow through interactive codebase chat' })).toBeVisible();
    });
    

    test('should handle button - Video demonstrating how GitHub', async ({ page }) => {
                await expect(page.getByRole('generic', { name: 'Video demonstrating how GitHub Copilot accelerates workflow through interactive codebase chat is currently paused. Click to play.' })).toBeVisible();
        await page.getByRole('generic', { name: 'Video demonstrating how GitHub Copilot accelerates workflow through interactive codebase chat is currently paused. Click to play.' }).click();
    });
    

    test('should handle video - GitHub Copilot model choices', async ({ page }) => {
                await expect(page.getByRole('generic', { name: 'GitHub Copilot model choices' })).toBeVisible();
    });
    

    test('should handle button - GitHub Copilot model choices i', async ({ page }) => {
                await expect(page.getByRole('generic', { name: 'GitHub Copilot model choices is currently paused. Click to play.' })).toBeVisible();
        await page.getByRole('generic', { name: 'GitHub Copilot model choices is currently paused. Click to play.' }).click();
    });
    

    test('should handle a - Back to content', async ({ page }) => {
                await expect(page.getByRole('generic', { name: 'Back to content' })).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByRole('generic', { name: 'Back to content' }).click()
        ]);
        await expect(page).toHaveURL('#footnote-ref-1-1');
    });
    

    test('should handle a - Go to GitHub homepage', async ({ page }) => {
                await expect(page.getByRole('generic', { name: 'Go to GitHub homepage' })).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByRole('generic', { name: 'Go to GitHub homepage' }).click()
        ]);
        await expect(page).toHaveURL('/');
    });
    

    test('should handle nav - Legal and Resource Links', async ({ page }) => {
                await expect(page.getByRole('generic', { name: 'Legal and Resource Links' })).toBeVisible();
    });
    

    test('should handle nav - GitHubs Social Media Links', async ({ page }) => {
                await expect(page.getByRole('generic', { name: 'GitHub's Social Media Links' })).toBeVisible();
    });
    
});
