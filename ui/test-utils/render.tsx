/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import type { ReactNode } from 'react';

import { ThemeProvider } from '@geti/ui/theme';
import { QueryClientProvider } from '@tanstack/react-query';
import { RenderOptions, render as rtlRender } from '@testing-library/react';
import { createMemoryRouter, RouterProvider } from 'react-router';

import { queryClient } from '../src/providers';
import { routes } from '../src/router';

interface Options extends RenderOptions {
    route: string;
    path: string;
}

const TestProviders = ({ children }: { children: ReactNode }) => {
    return (
        <QueryClientProvider client={queryClient}>
            <ThemeProvider>{children}</ThemeProvider>
        </QueryClientProvider>
    );
};

export const render = (
    ui: ReactNode,
    options: Options = { route: routes.project({ projectId: '1' }), path: routes.project.pattern }
) => {
    const router = createMemoryRouter(
        [
            {
                path: options.path,
                element: <TestProviders>{ui}</TestProviders>,
            },
        ],
        {
            initialEntries: [options.route],
            initialIndex: 0,
        }
    );

    return rtlRender(<RouterProvider router={router} />);
};
