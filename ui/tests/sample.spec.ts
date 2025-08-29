import { expect, test } from '@playwright/test';

test.describe('Sample test - to be removed', () => {
  test('Check if main components are properly shown', async ({ page }) => {
    await page.goto('/');

    await expect(page.getByText('Geti Prompt')).toBeVisible();
  });
});
