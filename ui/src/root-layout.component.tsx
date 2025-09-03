/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */
import { Grid, minmax, View } from '@geti/ui';
import { Header } from './components/header.component';

export const RootLayout = () => {
  return (
    <Grid
      areas={[
        'header header header',
        'toolbar prompt-sidebar sidebar',
        'main prompt-sidebar sidebar',
        'main-button prompt-sidebar sidebar',
      ]}
      rows={['size-800', 'size-700', '1fr', 'size-600']}
      columns={[minmax('60%', '1fr'), minmax(0, '37.5%'), 'size-600']}
      height={'100vh'}
    >
      <Header />

      <View gridArea={'toolbar'} borderColor={'blue-400'} borderWidth={'thin'}>
        Toolbar
      </View>

      <View
        gridArea={'prompt-sidebar'}
        borderColor={'chartreuse-400'}
        borderWidth={'thin'}
      >
        Primary toolbar
      </View>

      <View
        gridArea={'sidebar'}
        borderColor={'celery-300'}
        borderWidth={'thin'}
      >
        Side
      </View>

      <View gridArea={'main'} borderColor={'red-400'} borderWidth={'thin'}>
        Main
      </View>

      <View
        gridArea={'main-button'}
        borderColor={'green-400'}
        borderWidth={'thin'}
      >
        Button
      </View>
    </Grid>
  );
};
