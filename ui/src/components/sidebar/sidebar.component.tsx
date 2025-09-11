/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from 'react';

import { Wand } from '@geti-prompt/icons';
import { Flex, Grid, minmax, ToggleButton, View } from '@geti/ui';

import { PromptSidebar } from '../../features/prompt-sidebar/prompt-sidebar.component';

import styles from './sidebar.module.scss';

export const Sidebar = () => {
    const [isPromptSidebarOpen, setIsPromptSidebarOpen] = useState<boolean>(true);
    const gridTemplateColumns = isPromptSidebarOpen
        ? 'clamp(var(--spectrum-global-dimension-size-4600), 35vw, 36rem) var(--spectrum-global-dimension-size-600)'
        : '0 var(--spectrum-global-dimension-size-600)';

    return (
        <Grid
            gridArea={'sidebar'}
            UNSAFE_className={styles.container}
            data-expanded={isPromptSidebarOpen}
            UNSAFE_style={{
                gridTemplateColumns,
            }}
        >
            <View gridColumn={'1/2'} UNSAFE_className={styles.promptSidebarContainer}>
                <PromptSidebar />
            </View>
            <View gridColumn={'2/3'} backgroundColor={'gray-200'} padding={'size-100'}>
                <Flex direction={'column'} height={'100%'} alignItems={'center'}>
                    <ToggleButton
                        isQuiet
                        isSelected={isPromptSidebarOpen}
                        onChange={setIsPromptSidebarOpen}
                        UNSAFE_className={styles.toggleButton}
                        aria-label='Toggle prompt sidebar'
                    >
                        <Wand />
                    </ToggleButton>
                </Flex>
            </View>
        </Grid>
    );
};
