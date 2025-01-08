
import { test, expect } from '@playwright/test';

test.describe('https://github.com/customer-stories/mercado-libre', () => {
    test.beforeEach(async ({ page }) => {
        await page.goto('https://github.com/customer-stories/mercado-libre');
    });
    
    
    test('should handle button - Close', async ({ page }) => {
                await expect(page.getByRole('generic', { name: 'Close' })).toBeVisible();
        await page.getByRole('generic', { name: 'Close' }).click();
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
    

    test('should handle a - Learn more', async ({ page }) => {
                await expect(page.getByText('Learn more')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Learn more').click()
        ]);
        await expect(page).toHaveURL('https://github.com/features/copilot/?utm_source=github&utm_medium=banner&utm_campaign=copilotfree-bannerheader-customer-stories-mercado-libre');
    });
    

    test('should handle a - Homepage', async ({ page }) => {
                await expect(page.getByRole('generic', { name: 'Homepage' })).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByRole('generic', { name: 'Homepage' }).click()
        ]);
        await expect(page).toHaveURL('https://github.com/');
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
        await expect(page).toHaveURL('https://github.com/login?return_to=https%3A%2F%2Fgithub.com%2Fcustomer-stories%2Fmercado-libre');
    });
    

    test('should handle a - Sign up', async ({ page }) => {
                await expect(page.getByText('Sign up')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Sign up').click()
        ]);
        await expect(page).toHaveURL('https://github.com/signup?ref_cta=Sign+up&ref_loc=header+logged+out&ref_page=%2Fcustomer-stories%2Fmercado-libre&source=header');
    });
    

    test('should handle a - Customer Stories', async ({ page }) => {
                await expect(page.getByText('Customer Stories')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Customer Stories').click()
        ]);
        await expect(page).toHaveURL('https://github.com/customer-stories');
    });
    

    test('should handle a - Enterprise', async ({ page }) => {
                await expect(page.getByText('Enterprise')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Enterprise').click()
        ]);
        await expect(page).toHaveURL('https://github.com/customer-stories/enterprise');
    });
    

    test('should handle a - Team', async ({ page }) => {
                await expect(page.getByText('Team')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Team').click()
        ]);
        await expect(page).toHaveURL('https://github.com/customer-stories/team');
    });
    

    test('should handle a - All stories', async ({ page }) => {
                await expect(page.getByText('All stories')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('All stories').click()
        ]);
        await expect(page).toHaveURL('https://github.com/customer-stories/all');
    });
    

    test('should handle a - Start a free trial', async ({ page }) => {
                await expect(page.getByText('Start a free trial')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Start a free trial').click()
        ]);
        await expect(page).toHaveURL('https://github.com/organizations/enterprise_plan?ref_cta=Start+a+free+trial&ref_loc=customer+stories+nav&ref_page=%2Fcustomer-stories%2Fmercado-libre');
    });
    

    test('should handle a - Play video
          Accelerat', async ({ page }) => {
                await expect(page.getByText('Play video
          Accelerating Commerce: Mercado Libre + GitHub')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Play video
          Accelerating Commerce: Mercado Libre + GitHub').click()
        ]);
        await expect(page).toHaveURL('https://www.youtube.com/watch?v=U-BTaIgpT5o');
    });
    

    test('should handle a - GitHub Enterprise', async ({ page }) => {
                await expect(page.getByText('GitHub Enterprise')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('GitHub Enterprise').click()
        ]);
        await expect(page).toHaveURL('https://github.com/enterprise');
    });
    

    test('should handle a - GitHub Copilot', async ({ page }) => {
                await expect(page.getByText('GitHub Copilot')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('GitHub Copilot').click()
        ]);
        await expect(page).toHaveURL('https://github.com/features/copilot');
    });
    

    test('should handle a - GitHub Advanced Security', async ({ page }) => {
                await expect(page.getByText('GitHub Advanced Security')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('GitHub Advanced Security').click()
        ]);
        await expect(page).toHaveURL('https://github.com/enterprise/advanced-security');
    });
    

    test('should handle a - free trial
                of ', async ({ page }) => {
                await expect(page.getByText('free trial
                of GitHub Enterprise')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('free trial
                of GitHub Enterprise').click()
        ]);
        await expect(page).toHaveURL('https://github.com/organizations/enterprise_plan?ref_cta=free+trial&ref_loc=Customer+story+Mercado+Libre&ref_page=%2Fcustomer-stories%2Fmercado-libre');
    });
    

    test('should handle a - other plans?
                f', async ({ page }) => {
                await expect(page.getByText('other plans?
                from GitHub')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('other plans?
                from GitHub').click()
        ]);
        await expect(page).toHaveURL('https://github.com/pricing?ref_cta=other+plans&ref_loc=Customer+story+Mercado+Libre&ref_page=%2Fcustomer-stories%2Fmercado-libre');
    });
    

    test('should handle a - The ReadME Project
    
    St', async ({ page }) => {
                await expect(page.getByText('The ReadME Project
    
    Stories and voices from the developer community.

    
      Learn more about The ReadME Project')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('The ReadME Project
    
    Stories and voices from the developer community.

    
      Learn more about The ReadME Project').click()
        ]);
        await expect(page).toHaveURL('https://github.com/readme');
    });
    

    test('should handle a - GitHub Copilot
    
    AI pai', async ({ page }) => {
                await expect(page.getByText('GitHub Copilot
    
    AI pair programmer that helps you write code faster.

    
      Learn more about GitHub Copilot')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('GitHub Copilot
    
    AI pair programmer that helps you write code faster.

    
      Learn more about GitHub Copilot').click()
        ]);
        await expect(page).toHaveURL('https://github.com/features/copilot');
    });
    

    test('should handle a - Executive Insights
    
    Ge', async ({ page }) => {
                await expect(page.getByText('Executive Insights
    
    Get expert perspectives. Stay ahead with insights from industry leaders.

    
      Learn more about Executive Insights')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Executive Insights
    
    Get expert perspectives. Stay ahead with insights from industry leaders.

    
      Learn more about Executive Insights').click()
        ]);
        await expect(page).toHaveURL('https://github.com/solutions/executive-insights');
    });
    

    test('should handle a - Create a free organization', async ({ page }) => {
                await expect(page.getByText('Create a free organization')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Create a free organization').click()
        ]);
        await expect(page).toHaveURL('https://github.com/account/organizations/new?plan=free&ref_cta=Create+a+free+organization&ref_loc=Footer+cards&ref_page=%2Fcustomer-stories%2Fmercado-libre');
    });
    

    test('should handle a - Continue with Team', async ({ page }) => {
                await expect(page.getByText('Continue with Team')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Continue with Team').click()
        ]);
        await expect(page).toHaveURL('https://github.com/join?plan=business&ref_cta=Continue+with+Team&ref_loc=Footer+Cards&ref_page=%2Fcustomer-stories%2Fmercado-libre');
    });
    

    test('should handle a - Enterprise', async ({ page }) => {
                await expect(page.getByText('Enterprise')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Enterprise').click()
        ]);
        await expect(page).toHaveURL('https://github.com/organizations/enterprise_plan?ref_cta=Enterprise&ref_loc=Footer+Cards&ref_page=%2Fcustomer-stories%2Fmercado-libre');
    });
    

    test('should handle a - Check out our plans for indivi', async ({ page }) => {
                await expect(page.getByText('Check out our plans for individuals')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Check out our plans for individuals').click()
        ]);
        await expect(page).toHaveURL('https://github.com/pricing');
    });
    

    test('should handle a - Go to GitHub homepage', async ({ page }) => {
                await expect(page.getByRole('generic', { name: 'Go to GitHub homepage' })).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByRole('generic', { name: 'Go to GitHub homepage' }).click()
        ]);
        await expect(page).toHaveURL('https://github.com/');
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
        await expect(page).toHaveURL('https://github.com/features');
    });
    

    test('should handle a - Enterprise', async ({ page }) => {
                await expect(page.getByText('Enterprise')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Enterprise').click()
        ]);
        await expect(page).toHaveURL('https://github.com/enterprise');
    });
    

    test('should handle a - Copilot', async ({ page }) => {
                await expect(page.getByText('Copilot')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Copilot').click()
        ]);
        await expect(page).toHaveURL('https://github.com/features/copilot');
    });
    

    test('should handle a - Security', async ({ page }) => {
                await expect(page.getByText('Security')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Security').click()
        ]);
        await expect(page).toHaveURL('https://github.com/security');
    });
    

    test('should handle a - Pricing', async ({ page }) => {
                await expect(page.getByText('Pricing')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Pricing').click()
        ]);
        await expect(page).toHaveURL('https://github.com/pricing');
    });
    

    test('should handle a - Team', async ({ page }) => {
                await expect(page.getByText('Team')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Team').click()
        ]);
        await expect(page).toHaveURL('https://github.com/team');
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
        await expect(page).toHaveURL('https://github.com/enterprise/premium-support');
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
        await expect(page).toHaveURL('https://github.com/customer-stories?type=enterprise');
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
        await expect(page).toHaveURL('https://github.com/readme');
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
        await expect(page).toHaveURL('https://github.com/newsroom');
    });
    

    test('should handle a - Inclusion', async ({ page }) => {
                await expect(page.getByText('Inclusion')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('Inclusion').click()
        ]);
        await expect(page).toHaveURL('https://github.com/about/diversity');
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
        await expect(page).toHaveURL('https://github.com/sitemap');
    });
    

    test('should handle a - What is Git?', async ({ page }) => {
                await expect(page.getByText('What is Git?')).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByText('What is Git?').click()
        ]);
        await expect(page).toHaveURL('https://github.com/git-guides');
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
    

    test('should handle button - Close', async ({ page }) => {
                await expect(page.getByRole('generic', { name: 'Close' })).toBeVisible();
        await page.getByRole('generic', { name: 'Close' }).click();
    });
    

    test('should handle a - Homepage', async ({ page }) => {
                await expect(page.getByRole('generic', { name: 'Homepage' })).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByRole('generic', { name: 'Homepage' }).click()
        ]);
        await expect(page).toHaveURL('https://github.com/');
    });
    

    test('should handle button - Search or jump to…', async ({ page }) => {
                await expect(page.getByRole('generic', { name: 'Search or jump to…' })).toBeVisible();
        await page.getByRole('generic', { name: 'Search or jump to…' }).click();
    });
    

    test('should handle a - Go to GitHub homepage', async ({ page }) => {
                await expect(page.getByRole('generic', { name: 'Go to GitHub homepage' })).toBeVisible();
        await Promise.all([
            page.waitForNavigation(),
            page.getByRole('generic', { name: 'Go to GitHub homepage' }).click()
        ]);
        await expect(page).toHaveURL('https://github.com/');
    });
    
});
