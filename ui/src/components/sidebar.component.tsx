/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Grid, View } from '@geti/ui';

import { PromptSidebar } from '../features/prompt-sidebar/prompt-sidebar.component';

export const Sidebar = () => {
    return (
        <Grid gridArea={'sidebar'} height={'100%'} columns={['1fr', 'size-600']}>
            <PromptSidebar />
            <View backgroundColor={'gray-200'}>Actions</View>
        </Grid>
    );
};
