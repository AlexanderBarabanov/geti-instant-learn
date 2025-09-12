/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ReactNode } from 'react';

import { Disclosure, DisclosurePanel, DisclosureTitle, Flex, Text } from '@geti/ui';
import { clsx } from 'clsx';

import styles from './disclosure-group.module.scss';

interface DisclosureGroupProps<Value extends string> {
    value: Value | null;
    onChange?: (value: Value) => void;
    items: { value: Value; label: string; icon: ReactNode; content?: ReactNode }[];
}

export const DisclosureGroup = <Value extends string>({ onChange, items, value }: DisclosureGroupProps<Value>) => {
    return (
        <Flex width={'100%'} direction={'column'} gap={'size-100'}>
            {items.map((item) => (
                <Disclosure
                    isQuiet
                    key={item.label}
                    UNSAFE_className={clsx(styles.disclosure, {
                        [styles.selected]: item.value === value,
                    })}
                    onExpandedChange={() => {
                        onChange !== undefined && onChange(item.value);
                    }}
                >
                    <DisclosureTitle UNSAFE_className={styles.disclosureTitleContainer}>
                        <Flex alignItems={'center'} justifyContent={'space-between'} width={'100%'}>
                            <Flex marginStart={'size-50'} alignItems={'center'} gap={'size-100'}>
                                {item.icon}
                                <Text UNSAFE_className={styles.disclosureTitle}>{item.label}</Text>
                            </Flex>
                        </Flex>
                    </DisclosureTitle>
                    <DisclosurePanel>{item.content}</DisclosurePanel>
                </Disclosure>
            ))}
        </Flex>
    );
};
