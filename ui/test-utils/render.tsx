/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import type { ReactNode } from 'react';

import { ThemeProvider } from '@geti/ui/theme';
import { QueryClientProvider } from '@tanstack/react-query';
import { RenderOptions, render as rtlRender } from '@testing-library/react';
import { MemoryRouter } from 'react-router';

import { queryClient } from '../src/providers';

export const render = (ui: ReactNode, options?: RenderOptions) => {
    return rtlRender(
        <QueryClientProvider client={queryClient}>
            <ThemeProvider>
                <MemoryRouter>{ui}</MemoryRouter>
            </ThemeProvider>
        </QueryClientProvider>,
        options
    );
};
