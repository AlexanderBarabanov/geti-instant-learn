/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Grid, minmax } from '@geti/ui';
import { Outlet } from 'react-router';

import { Header } from './components/header.component';
import { Sidebar } from './components/sidebar.component';
import { Toolbar } from './components/toolbar.component';

export const RootLayout = () => {
    return (
        <Grid
            areas={['header header header', 'toolbar prompt-sidebar sidebar', 'main prompt-sidebar sidebar']}
            rows={['size-800', 'size-700', '1fr']}
            columns={[minmax('60%', '1fr'), 'auto']}
            height={'100vh'}
        >
            <Header />

            <Toolbar />

            <Outlet />

            <Sidebar />
        </Grid>
    );
};
