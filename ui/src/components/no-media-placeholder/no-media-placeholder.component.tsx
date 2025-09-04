/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Content, Flex, View } from '@adobe/react-spectrum';
import { NoMedia } from '@geti-prompt/icons';

import styles from './no-media-placeholder.module.scss';

interface NoMediaPlaceholderProps {
    title: string;
}

export const NoMediaPlaceholder = ({ title }: NoMediaPlaceholderProps) => {
    return (
        <View backgroundColor={'gray-400'} height={'100%'}>
            <Flex height={'100%'} width={'100%'} justifyContent={'center'} alignItems={'center'}>
                <Flex direction={'column'} gap={'size-100'} alignItems={'center'}>
                    <NoMedia />
                    <Content UNSAFE_className={styles.title}>{title}</Content>
                </Flex>
            </Flex>
        </View>
    );
};
