/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { NoMedia } from '@geti-prompt/icons';
import { Content, Flex, View } from '@geti/ui';

import styles from './no-media-placeholder.module.scss';

interface NoMediaPlaceholderProps {
    title: string;
}

export const NoMediaPlaceholder = ({ title }: NoMediaPlaceholderProps) => {
    return (
        <View backgroundColor={'gray-200'} height={'100%'} UNSAFE_className={styles.container}>
            <Flex height={'100%'} width={'100%'} justifyContent={'center'} alignItems={'center'}>
                <Flex direction={'column'} gap={'size-100'} alignItems={'center'}>
                    <NoMedia />
                    <Content UNSAFE_className={styles.title}>{title}</Content>
                </Flex>
            </Flex>
        </View>
    );
};
