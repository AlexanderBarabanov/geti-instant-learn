/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Grid, View } from '@geti/ui';

export const Sidebar = () => {
    return (
        <Grid gridArea={'sidebar'} height={'100%'} columns={['1fr', 'size-600']}>
            <View>Prompt</View>
            <View backgroundColor={'gray-200'}>Actions</View>
        </Grid>
    );
};
