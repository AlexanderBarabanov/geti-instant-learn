/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Divider, Flex, Heading, View } from '@geti/ui';

import { PromptModes } from './prompt-modes/prompt-modes.component';

export const PromptSidebar = () => {
    return (
        <View
            width={'30vw'}
            backgroundColor={'gray-100'}
            padding={'size-300'}
            height={'100%'}
            UNSAFE_style={{ boxSizing: 'border-box' }}
        >
            <Flex direction={'column'} height={'100%'}>
                <Heading margin={0}>Prompt</Heading>
                <View flex={1} padding={'size-300'} UNSAFE_style={{ boxSizing: 'border-box' }}>
                    <Flex direction={'column'} height={'100%'}>
                        <PromptModes />

                        <Divider size={'S'} marginY={'size-300'} />
                    </Flex>
                </View>
            </Flex>
        </View>
    );
};
